"""
Графический интерфейс Movie Shorts Pipeline (tkinter, без внешних зависимостей).

Схема работы: выбрать файл (или вставить ссылки Twitch) → название и год
подставляются сами → «Обработать» → все этапы проходят в фоне, лог и прогресс
видны в окне → в конце открывается запрос «Опубликовать в Telegram?».

Режимы:
- Фильм — субтитры → Gemini ищет вирусные сцены → вертикальные клипы
- Стрим — субтитры → Gemini ищет яркие моменты (или моменты задаются вручную)
- Twitch — клипы/VOD скачиваются по ссылкам и нарезаются как стрим
- Клип — локальный файл или ссылка на любое видео → вертикальный шортс

Usage:
    python gui.py
"""

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ai_engine import AIEngine
from config import Config
from core import (
    SOURCE_CLIP,
    SOURCE_MOVIE,
    SOURCE_STREAM,
    PipelineError,
    detect_year,
    guess_title,
    parse_moments,
    post_link,
    process_clip,
    process_stream,
    process_twitch_clips,
    process_video,
    publish_result,
    publish_youtube,
    rerender_clip,
)
from youtube_publisher import YouTubePublisher, SETUP_INSTRUCTION


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".webm", ".wmv", ".flv", ".ts"}


class PipelineGUI:
    """Окно приложения."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Movie Shorts Pipeline")
        root.geometry("860x640")
        root.minsize(720, 520)

        self.result = None          # MovieResult после успешной обработки
        self.busy = False
        self.events = queue.Queue() # из рабочего потока в главный
        self._clips_before_clip = None  # значение «Клипов:» до входа в режим клипа
        self._clip_segments = {}    # iid строки дерева → номер сегмента источника (1-based)

        self._build_ui()
        self._poll_events()

    # ---------- интерфейс ----------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        form = ttk.Frame(self.root)
        form.pack(fill="x", **pad)

        # Путь к файлу
        ttk.Label(form, text="Файл:").grid(row=0, column=0, sticky="w")
        self.path_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.path_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(form, text="Обзор...", command=self._choose_file).grid(
            row=0, column=2, sticky="e"
        )

        # Название
        ttk.Label(form, text="Название:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.title_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.title_var).grid(
            row=1, column=1, sticky="ew", padx=6, pady=(6, 0)
        )
        ttk.Label(form, text="(определяется на русском через Gemini, можно править)").grid(
            row=1, column=2, sticky="w", pady=(6, 0)
        )

        # Год и количество клипов
        opts = ttk.Frame(form)
        opts.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Label(opts, text="Год:").pack(side="left")
        self.year_var = tk.StringVar()
        ttk.Entry(opts, textvariable=self.year_var, width=8).pack(side="left", padx=(4, 16))
        ttk.Label(opts, text="(пусто = определить автоматически)").pack(side="left")

        ttk.Label(opts, text="Клипов:").pack(side="left", padx=(24, 0))
        self.clips_var = tk.IntVar(value=3)
        ttk.Spinbox(opts, from_=1, to=10, textvariable=self.clips_var, width=6).pack(
            side="left", padx=4
        )

        # Режим обработки и ручные моменты для стримов
        mode = ttk.Frame(form)
        mode.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        ttk.Label(mode, text="Режим:").pack(side="left")
        self.source_var = tk.StringVar(value="Фильм")
        source_box = ttk.Combobox(
            mode, textvariable=self.source_var,
            values=("Фильм", "Стрим", "Клип"), state="readonly", width=10,
        )
        source_box.pack(side="left", padx=(4, 16))
        source_box.bind("<<ComboboxSelected>>", lambda _e: self._on_source_changed())

        ttk.Label(mode, text="Моменты:").pack(side="left")
        self.moments_var = tk.StringVar()
        self.moments_entry = ttk.Entry(mode, textvariable=self.moments_var)
        self.moments_entry.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Label(mode, text="(ЧЧ:ММ:СС-ЧЧ:ММ:СС, …)").pack(side="left")

        # Ссылки Twitch: клип или VOD скачиваются автоматически
        twitch = ttk.Frame(form)
        twitch.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        # В режиме клипа поле принимает любой URL видео, поэтому подпись меняется
        self.twitch_label_var = tk.StringVar(value="Twitch:")
        ttk.Label(twitch, textvariable=self.twitch_label_var).pack(side="left")
        self.twitch_var = tk.StringVar()
        self.twitch_entry = ttk.Entry(twitch, textvariable=self.twitch_var)
        self.twitch_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.twitch_hint_var = tk.StringVar(
            value="(ссылка на клип/VOD, можно несколько через запятую)"
        )
        ttk.Label(twitch, textvariable=self.twitch_hint_var).pack(side="left")

        # Настройки монтажа: FPS, стиль субтитров и пресет
        settings = ttk.Frame(form)
        settings.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Label(settings, text="FPS:").pack(side="left")
        self.fps_var = tk.StringVar(value="по умолчанию")
        ttk.Combobox(
            settings, textvariable=self.fps_var,
            values=("по умолчанию", "30", "60"), state="readonly", width=13,
        ).pack(side="left", padx=(4, 16))

        ttk.Label(settings, text="Субтитры:").pack(side="left")
        self.subtitle_var = tk.StringVar(value="по умолчанию")
        ttk.Combobox(
            settings, textvariable=self.subtitle_var,
            values=("по умолчанию", "bold", "boxed", "karaoke"),
            state="readonly", width=13,
        ).pack(side="left", padx=(4, 16))

        ttk.Label(settings, text="Пресет:").pack(side="left")
        self.preset_var = tk.StringVar(value="по умолчанию")
        ttk.Combobox(
            settings, textvariable=self.preset_var,
            values=(
                "по умолчанию", "auto", "movie", "stream",
                "clip", "podcast", "gaming", "off",
            ),
            state="readonly", width=13,
        ).pack(side="left", padx=4)

        self._on_source_changed()

        form.columnconfigure(1, weight=1)

        # Кнопки запуска
        actions = ttk.Frame(self.root)
        actions.pack(fill="x", **pad)

        self.run_btn = ttk.Button(actions, text="Обработать", command=self._start_run)
        self.run_btn.pack(side="left")

        self.open_folder_btn = ttk.Button(
            actions, text="Открыть папку фильма", command=self._open_folder, state="disabled"
        )
        self.open_folder_btn.pack(side="left", padx=8)

        self.publish_btn = ttk.Button(
            actions, text="Опубликовать в Telegram", command=self._ask_publish, state="disabled"
        )
        self.publish_btn.pack(side="left")

        self.youtube_btn = ttk.Button(
            actions, text="Выложить в YouTube", command=self._ask_youtube, state="disabled"
        )
        self.youtube_btn.pack(side="left", padx=8)

        # Прогресс
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=10)

        # Лог и описание — две вкладки
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, **pad)

        self.log_text = tk.Text(self.notebook, height=18, wrap="word", state="disabled")
        self.notebook.add(self.log_text, text="Ход работы")

        self.desc_text = tk.Text(self.notebook, height=18, wrap="word")
        self.notebook.add(self.desc_text, text="Описание публикации")

        log_scroll = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        # Готовые клипы: список с длительностями и действиями над каждым
        clips_frame = ttk.Frame(self.notebook)
        self.notebook.add(clips_frame, text="Клипы")

        tree_wrap = ttk.Frame(clips_frame)
        tree_wrap.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        self.clips_tree = ttk.Treeview(
            tree_wrap, columns=("file", "duration"), show="headings"
        )
        self.clips_tree.heading("file", text="Файл")
        self.clips_tree.heading("duration", text="Длительность")
        self.clips_tree.column("file", anchor="w")
        self.clips_tree.column("duration", width=110, anchor="e", stretch=False)
        self.clips_tree.pack(side="left", fill="both", expand=True)

        tree_scroll = ttk.Scrollbar(
            tree_wrap, orient="vertical", command=self.clips_tree.yview
        )
        tree_scroll.pack(side="right", fill="y")
        self.clips_tree.configure(yscrollcommand=tree_scroll.set)

        clip_buttons = ttk.Frame(clips_frame)
        clip_buttons.pack(fill="x", padx=10, pady=6)
        ttk.Button(clip_buttons, text="Открыть", command=self._open_selected_clip).pack(
            side="left"
        )
        ttk.Button(
            clip_buttons, text="Отклонить", command=self._reject_selected_clip
        ).pack(side="left", padx=8)
        ttk.Button(
            clip_buttons, text="Пересоздать", command=self._rerender_selected_clip
        ).pack(side="left")

        # Статусная строка
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w").pack(
            fill="x", side="bottom"
        )

    # ---------- ввод ----------

    def _choose_file(self):
        start_dir = Path(self.path_var.get()).parent if self.path_var.get() else None
        initial = str(start_dir) if start_dir and start_dir.exists() else ""

        path = filedialog.askopenfilename(
            title="Выберите файл фильма",
            initialdir=initial,
            filetypes=[
                ("Видео", " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return

        self.path_var.set(path)

        # Название подставляется из имени файла, год определяется сам
        title_guess = guess_title(path)
        self.title_var.set(title_guess)

        year = detect_year(path)
        self.year_var.set(str(year) if year else "")

        self._log(f"Выбран файл: {path}")
        self._log(
            f"Определено автоматически — название: {title_guess!r}, "
            f"год: {year or 'не найден'}"
        )
        # Русский перевод названия нужен только фильмам; стримы и клипы Twitch
        # называются как есть
        if self.source_var.get() == "Фильм":
            self._resolve_title_async(title_guess, year)

    def _resolve_title_async(self, raw_title: str, year):
        """Запрос русского названия в фоне: GUI не блокируется на ответе Gemini."""
        if not raw_title:
            return

        self.status_var.set("Определяем русское название...")
        worker = threading.Thread(
            target=self._resolve_title_worker,
            args=(raw_title, year),
            daemon=True,
        )
        worker.start()

    def _resolve_title_worker(self, raw_title: str, year):
        """Рабочий поток определения русского названия через Gemini."""
        try:
            engine = AIEngine(Config())
            resolved = engine.resolve_title(raw_title, year)
        except Exception as e:  # noqa: BLE001 — ошибка перевода не критична
            self.events.put(("title_resolved", raw_title, None, str(e)))
        else:
            self.events.put(("title_resolved", raw_title, resolved, None))

    def _on_source_changed(self):
        """Моменты активны только в стриме; в клипе поле ссылки принимает любой URL."""
        mode = self.source_var.get()
        is_stream = mode == "Стрим"
        is_clip = mode == "Клип"

        self.moments_entry.config(state="normal" if is_stream else "disabled")
        if not is_stream:
            self.moments_var.set("")

        self.twitch_label_var.set("Файл / ссылка:" if is_clip else "Twitch:")
        self.twitch_hint_var.set(
            "(ссылка на любое видео или путь к файлу, можно несколько через запятую)"
            if is_clip
            else "(ссылка на клип/VOD, можно несколько через запятую)"
        )

        # В режиме клипа по умолчанию один короткий ролик; вернувшись
        # в другие режимы, восстанавливаем прежнее значение
        if is_clip:
            if self._clips_before_clip is None:
                self._clips_before_clip = self.clips_var.get()
                self.clips_var.set(1)
        elif self._clips_before_clip is not None:
            self.clips_var.set(self._clips_before_clip)
            self._clips_before_clip = None

    def _validated_inputs(self):
        path = self.path_var.get().strip()
        title = self.title_var.get().strip()
        raw_year = self.year_var.get().strip()
        clips = self.clips_var.get()
        mode_label = self.source_var.get()
        if mode_label == "Стрим":
            source = SOURCE_STREAM
        elif mode_label == "Клип":
            source = SOURCE_CLIP
        else:
            source = SOURCE_MOVIE
        twitch_raw = self.twitch_var.get().strip()
        moments_raw = self.moments_var.get().strip() if source == SOURCE_STREAM else ""

        twitch_urls = [u for u in re.split(r"[,\s]+", twitch_raw) if u] if twitch_raw else []

        clip_inputs = []
        if source == SOURCE_CLIP:
            # Либо локальный файл, либо ссылка (или несколько ссылок) на видео
            if path and twitch_urls:
                messagebox.showwarning(
                    "Лишний источник",
                    "Укажите либо файл, либо ссылку — не одновременно.",
                )
                return None
            if path:
                if not Path(path).exists():
                    messagebox.showerror("Файл не найден", f"Такого файла нет:\n{path}")
                    return None
                clip_inputs = [path]
            else:
                clip_inputs = twitch_urls
            if not clip_inputs:
                messagebox.showwarning(
                    "Нет источника", "Выберите файл или укажите ссылку на видео."
                )
                self._choose_file()
                return None
            # Название в режиме клипа необязательно: берётся из заголовка видео
        else:
            if not twitch_urls:
                if not path:
                    messagebox.showwarning("Нет файла", "Выберите файл или укажите ссылку Twitch.")
                    self._choose_file()
                    return None
                if not Path(path).exists():
                    messagebox.showerror("Файл не найден", f"Такого файла нет:\n{path}")
                    return None

            # Название обязательно, кроме клипов Twitch — там берётся из заголовка
            if not title and not twitch_urls:
                messagebox.showwarning("Нет названия", "Укажите название.")
                return None

        year = None
        if raw_year:
            if not raw_year.isdigit():
                messagebox.showwarning("Год", f"Год должен быть числом: {raw_year!r}")
                return None
            year = int(raw_year)

        moments = ""
        if moments_raw:
            try:
                parse_moments(moments_raw)
            except PipelineError as e:
                messagebox.showwarning("Моменты", str(e))
                return None
            moments = moments_raw

        # Настройки монтажа: «по умолчанию» означает «берутся из .env» (None)
        render_kwargs = self._render_kwargs()

        return {
            "path": path,
            "title": title,
            "year": year,
            "clips": clips,
            "source": source,
            "twitch_urls": twitch_urls,
            "moments": moments,
            "clip_inputs": clip_inputs,
            "fps": render_kwargs["fps"],
            "subtitle_style": render_kwargs["subtitle_style"],
            "preset": render_kwargs["preset"],
        }

    # ---------- запуск ----------

    def _start_run(self):
        if self.busy:
            return

        inputs = self._validated_inputs()
        if not inputs:
            return

        self.result = None
        self._set_busy(True, "Обработка началась...")
        self._clear_log()
        self.progress["value"] = 0
        self.clips_tree.delete(*self.clips_tree.get_children())
        self._log("=" * 60)
        self._log("Запуск обработки")
        if inputs["source"] == SOURCE_CLIP:
            self._log(f"Режим: Клип ({len(inputs['clip_inputs'])} источник(ов))")
        elif inputs["twitch_urls"]:
            self._log(f"Режим: Twitch ({len(inputs['twitch_urls'])} ссылок)")
        else:
            self._log(f"Режим: {'Стрим' if inputs['source'] == SOURCE_STREAM else 'Фильм'}")
        if inputs["twitch_urls"] and (
            inputs["fps"] or inputs["subtitle_style"] or inputs["preset"]
        ):
            self._log(
                "Примечание: FPS/субтитры/пресет к пачке клипов Twitch не применяются "
                "(для неё настройки берутся из .env)."
            )
        self._log("=" * 60)

        worker = threading.Thread(
            target=self._run_worker,
            args=(inputs,),
            daemon=True,
        )
        worker.start()

    def _run_worker(self, inputs):
        """Рабочий поток: только складывает события в очередь, GUI не трогает."""
        log_cb = lambda message: self.events.put(("log", message))
        progress_cb = lambda done, total: self.events.put(("progress", done, total))
        # Настройки монтажа передаём, только если пользователь их выбрал
        # («по умолчанию» → None → ядро берёт значения из .env)
        render_kwargs = {}
        if inputs["fps"]:
            render_kwargs["fps"] = inputs["fps"]
        if inputs["subtitle_style"]:
            render_kwargs["subtitle_style"] = inputs["subtitle_style"]
        if inputs["preset"]:
            render_kwargs["preset"] = inputs["preset"]

        try:
            if inputs["source"] == SOURCE_CLIP:
                # Один вызов на источник: путь или ссылка, каждый даёт свой результат.
                # Несколько ссылок собираем в один результат: клипы — все вместе,
                # пост и описание — по первому источнику.
                result = None
                for item in inputs["clip_inputs"]:
                    log_cb(f"Источник: {item}")
                    item_result = process_clip(
                        source_path_or_url=item,
                        title=inputs["title"] or None,
                        clips_count=inputs["clips"],
                        env_path=None,
                        log=log_cb,
                        progress=progress_cb,
                        **render_kwargs,
                    )
                    if result is None:
                        result = item_result
                    else:
                        result.clip_paths.extend(item_result.clip_paths)
            elif inputs["twitch_urls"]:
                result = process_twitch_clips(
                    urls=inputs["twitch_urls"],
                    title=inputs["title"] or None,
                    year=inputs["year"],
                    log=log_cb,
                    progress=progress_cb,
                )
            elif inputs["source"] == SOURCE_STREAM:
                result = process_stream(
                    input_path=inputs["path"],
                    title=inputs["title"],
                    clips_count=inputs["clips"],
                    moments=inputs["moments"],
                    year=inputs["year"],
                    log=log_cb,
                    progress=progress_cb,
                    **render_kwargs,
                )
            else:
                result = process_video(
                    input_path=inputs["path"],
                    title=inputs["title"],
                    clips_count=inputs["clips"],
                    year=inputs["year"],
                    log=log_cb,
                    progress=progress_cb,
                    **render_kwargs,
                )
        except PipelineError as e:
            self.events.put(("run_failed", str(e)))
        except Exception as e:  # noqa: BLE001 - показываем любое неожиданное
            logger.exception("Непредвиденная ошибка обработки")
            self.events.put(("run_failed", f"{type(e).__name__}: {e}"))
        else:
            self.events.put(("run_done", result))

    def _ask_publish(self):
        if self.result is None:
            return
        if not messagebox.askyesno(
            "Публикация",
            f"Опубликовать пост о фильме «{self.result.title}» "
            f"(код {self.result.code}) в Telegram-канале?",
        ):
            self._log("Публикация отменена пользователем.")
            self.status_var.set("Публикация отменена")
            return

        self._set_busy(True, "Публикация...")
        worker = threading.Thread(target=self._publish_worker, daemon=True)
        worker.start()

    def _publish_worker(self):
        result = self.result
        try:
            publish_result(result, log=lambda message: self.events.put(("log", message)))
        except PipelineError as e:
            self.events.put(("publish_failed", str(e)))
        except Exception as e:  # noqa: BLE001
            logger.exception("Непредвиденная ошибка публикации")
            self.events.put(("publish_failed", f"{type(e).__name__}: {e}"))
        else:
            self.events.put(("publish_done", result))

    def _ask_youtube(self):
        if self.result is None or self.busy:
            return

        if not YouTubePublisher(Config()).is_configured():
            messagebox.showinfo("YouTube не настроен", SETUP_INSTRUCTION)
            return

        clips_count = len(self.result.clip_paths)
        if not messagebox.askyesno(
            "YouTube",
            f"Загрузить {clips_count} {clips_count in (1,) and 'клип' or 'клипов'} "
            f"фильма «{self.result.title}» в YouTube Shorts?",
        ):
            return

        self._log("Откроется браузер Google, если вход в YouTube ещё не выполнен.")
        self._set_busy(True, "Загрузка в YouTube...")
        worker = threading.Thread(target=self._youtube_worker, daemon=True)
        worker.start()

    def _youtube_worker(self):
        result = self.result
        try:
            uploaded = publish_youtube(
                result, log=lambda message: self.events.put(("log", message))
            )
        except PipelineError as e:
            self.events.put(("youtube_failed", str(e)))
        except Exception as e:  # noqa: BLE001
            logger.exception("Непредвиденная ошибка загрузки в YouTube")
            self.events.put(("youtube_failed", f"{type(e).__name__}: {e}"))
        else:
            self.events.put(("youtube_done", uploaded))

    def _open_folder(self):
        if self.result is None:
            return
        folder = str(self.result.movie_dir)
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except OSError as e:
            messagebox.showerror("Папка", f"Не удалось открыть {folder}:\n{e}")

    # ---------- готовые клипы ----------

    def _populate_clips_tree(self):
        """Список готовых клипов с длительностями в момент завершения обработки."""
        self.clips_tree.delete(*self.clips_tree.get_children())
        self._clip_segments.clear()
        for index, clip_path in enumerate(self.result.clip_paths, start=1):
            # Номер клипа нужен пересозданию: core ждёт 1-based индекс сегмента
            self._clip_segments[str(clip_path)] = index
            duration = self._clip_duration(clip_path)
            self.clips_tree.insert(
                "", "end",
                iid=str(clip_path),  # путь как идентификатор строки
                values=(Path(clip_path).name, self._format_duration(duration)),
            )

    @staticmethod
    def _clip_duration(clip_path) -> Optional[float]:
        """Длительность готового клипа через ffprobe; None, если не удалось."""
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    str(clip_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            duration = float(json.loads(proc.stdout).get("format", {}).get("duration", 0))
            return duration or None
        except Exception:  # noqa: BLE001 — отсутствие длительности не критично
            return None

    @staticmethod
    def _format_duration(duration: Optional[float]) -> str:
        return f"{int(duration // 60)}:{int(duration % 60):02d}" if duration else "—"

    def _selected_clip(self) -> Optional[str]:
        selection = self.clips_tree.selection()
        if not selection:
            messagebox.showinfo("Клипы", "Выберите клип в списке.")
            return None
        return selection[0]

    def _open_selected_clip(self):
        """Открыть выбранный клип системным проигрывателем."""
        clip_path = self._selected_clip()
        if clip_path is None:
            return
        if not Path(clip_path).exists():
            messagebox.showerror("Клип не найден", f"Такого файла нет:\n{clip_path}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(clip_path)  # noqa: S606 — только локальные файлы
            elif sys.platform == "darwin":
                subprocess.Popen(["open", clip_path])
            else:
                subprocess.Popen(["xdg-open", clip_path])
        except OSError as e:
            messagebox.showerror("Клип", f"Не удалось открыть {clip_path}:\n{e}")

    def _reject_selected_clip(self):
        """Удалить файл клипа и убрать его из результата публикации."""
        clip_path = self._selected_clip()
        if clip_path is None:
            return
        if not messagebox.askyesno(
            "Отклонить клип",
            f"Удалить файл без возможности восстановления?\n\n{clip_path}",
        ):
            return

        try:
            os.remove(clip_path)
        except OSError as e:
            messagebox.showerror("Отклонение клипа", f"Не удалось удалить файл:\n{e}")
            return

        if self.result is not None:
            self.result.clip_paths = [
                p for p in self.result.clip_paths if str(p) != clip_path
            ]
        self.clips_tree.delete(clip_path)
        self._clip_segments.pop(clip_path, None)
        self._log(f"Клип отклонён: {Path(clip_path).name}")

    def _render_kwargs(self) -> dict:
        """Настройки монтажа из строки настроек; «по умолчанию» → None → из .env."""
        fps_raw = self.fps_var.get().strip()
        subtitle_raw = self.subtitle_var.get().strip()
        preset_raw = self.preset_var.get().strip()
        return {
            "fps": int(fps_raw) if fps_raw.isdigit() else None,
            "subtitle_style": subtitle_raw if subtitle_raw != "по умолчанию" else None,
            "preset": preset_raw if preset_raw != "по умолчанию" else None,
        }

    def _rerender_selected_clip(self):
        """Пересоздать тот же отрезок источника свежим рендером (в фоне)."""
        clip_path = self._selected_clip()
        if clip_path is None:
            return
        clip_index = self._clip_segments.get(clip_path)
        if not clip_index or self.result is None \
                or clip_index > len(self.result.clip_segments):
            messagebox.showinfo(
                "Пересоздание",
                "Для этого клипа нет данных об отрезке источника — "
                "пересоздание недоступно.",
            )
            return

        self._set_busy(True, "Пересоздание клипа...")
        worker = threading.Thread(
            target=self._rerender_worker,
            args=(clip_path, clip_index, self._render_kwargs()),
            daemon=True,
        )
        worker.start()

    def _rerender_worker(self, clip_path, clip_index, render_kwargs):
        """Рабочий поток пересоздания: тот же сегмент, свежий рендер."""
        try:
            info = rerender_clip(
                self.result,
                clip_index,
                log=lambda message: self.events.put(("log", message)),
                **render_kwargs,
            )
        except PipelineError as e:
            self.events.put(("rerender_failed", clip_path, str(e)))
        except Exception as e:  # noqa: BLE001 - показываем любое неожиданное
            logger.exception("Непредвиденная ошибка пересоздания")
            self.events.put(("rerender_failed", clip_path, f"{type(e).__name__}: {e}"))
        else:
            self.events.put(("rerender_done", clip_path, info))

    # ---------- обработка событий из рабочих потоков ----------

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_events)

    def _handle_event(self, event):
        kind = event[0]

        if kind == "log":
            self._log(event[1])
        elif kind == "title_resolved":
            self._on_title_resolved(event[1], event[2], event[3])
        elif kind == "progress":
            _, done, total = event
            self.progress["maximum"] = max(total, 1)
            self.progress["value"] = done
        elif kind == "run_done":
            self._on_run_done(event[1])
        elif kind == "run_failed":
            self._set_busy(False, "Ошибка обработки")
            self._log(f"ОШИБКА: {event[1]}")
            messagebox.showerror("Ошибка обработки", event[1])
        elif kind == "rerender_done":
            self._on_rerender_done(event[1], event[2])
        elif kind == "rerender_failed":
            self._set_busy(False, "Ошибка пересоздания")
            self._log(f"ОШИБКА: {event[2]}")
            messagebox.showerror("Пересоздание клипа", event[2])
        elif kind == "publish_done":
            self._on_publish_done(event[1])
        elif kind == "publish_failed":
            self._set_busy(False, "Ошибка публикации")
            self._log(f"ОШИБКА: {event[1]}")
            messagebox.showerror("Ошибка публикации", event[1])
        elif kind == "youtube_done":
            self._on_youtube_done(event[1])
        elif kind == "youtube_failed":
            self._set_busy(False, "Ошибка загрузки в YouTube")
            self._log(f"ОШИБКА: {event[1]}")
            messagebox.showerror("Ошибка YouTube", event[1])

    def _on_title_resolved(self, raw_title: str, resolved, error):
        """Результат фонового определения русского названия."""
        if error:
            self._log(
                f"Русское название определить не удалось: {error}. "
                f"Оставляю «{raw_title}» — попробую ещё раз при обработке."
            )
            self.status_var.set("Название не определено")
            return

        # Не затираем правку пользователя: меняем поле, только если там всё ещё догадка из имени файла
        if self.title_var.get().strip() == raw_title.strip():
            self.title_var.set(resolved)

        self._log(f"Русское название: «{resolved}»")
        self.status_var.set("Название определено")

    def _on_run_done(self, result):
        self.result = result
        self._set_busy(False, "Обработка завершена — можно публиковать")
        self.progress["value"] = self.progress["maximum"]
        self._populate_clips_tree()

        self.desc_text.delete("1.0", "end")
        self.desc_text.insert("1.0", result.description)
        self.notebook.select(self.desc_text)

        self.open_folder_btn.config(state="normal")
        self.publish_btn.config(state="normal")
        self.youtube_btn.config(state="normal")

        self._log("=" * 60)
        self._log("✅ Обработка завершена!")
        self._log("=" * 60)
        if result.source == SOURCE_STREAM and "streamer" in result.post_data:
            # Пачка клипов Twitch: код — внутренний идентификатор, не показываем
            streamer = result.post_data.get("streamer")
            self._log(f"Стример: {streamer or 'не определён'}")
        else:
            self._log(f"Код фильма: {result.code}")
        self._log(f"Создано клипов: {len(result.clip_paths)}")
        self._log(f"Постер: {result.poster_path}")
        self._log(f"Описание: {result.description_path}")
        self._log(f"Папка фильма: {result.movie_dir}")
        self._log("\nПроверьте описание и нажмите «Опубликовать в Telegram», если всё устраивает.")

        # Второй вопрос: сразу выложить клипы в YouTube Shorts
        if messagebox.askyesno(
            "YouTube",
            "Загрузить клипы фильма в YouTube Shorts?",
        ):
            self._ask_youtube()

    def _on_publish_done(self, result):
        self._set_busy(False, "Пост опубликован")
        self.publish_btn.config(state="disabled")
        self.youtube_btn.config(state="normal")
        self._log("=" * 60)
        self._log(f"✅ Пост опубликован! Message ID: {result.message_id}")
        link = post_link(result)
        if link:
            self._log(f"Просмотреть: {link}")
        messagebox.showinfo(
            "Опубликовано",
            f"Пост о фильме «{result.title}» опубликован.\n\n{link or ''}",
        )

    def _on_youtube_done(self, uploaded):
        self._set_busy(False, "Клипы загружены в YouTube")
        self._log("=" * 60)
        self._log(f"✅ В YouTube загружено клипов: {len(uploaded)}")
        for item in uploaded:
            self._log(f"  {Path(item['clip']).name} → {item['url']}")

        messagebox.showinfo(
            "YouTube",
            f"Загружено клипов: {len(uploaded)}\n\n"
            + "\n".join(item["url"] for item in uploaded),
        )

    def _on_rerender_done(self, old_clip_path, info):
        """Свежий рендер готов: обновляем строку дерева и путь в результате."""
        self._set_busy(False, "Клип пересоздан")
        new_path = str(info.get("clip_path") or old_clip_path)
        duration = info.get("duration") or self._clip_duration(new_path)

        if new_path != old_clip_path:
            # Рендер записал файл под другим именем: переносим строку,
            # путь в result.clip_paths уже обновлён rerender_clip
            self.clips_tree.delete(old_clip_path)
            self.clips_tree.insert(
                "", "end",
                iid=new_path,
                values=(Path(new_path).name, self._format_duration(duration)),
            )
            self._clip_segments[new_path] = self._clip_segments.pop(old_clip_path, 0)
        else:
            # Тот же путь: обновляем только длительность
            self.clips_tree.item(old_clip_path, values=(
                Path(old_clip_path).name, self._format_duration(duration),
            ))
        self._log(f"✅ Клип пересоздан: {Path(new_path).name}")

    # ---------- вспомогательное ----------

    def _set_busy(self, busy: bool, status: str = ""):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.run_btn.config(state=state, text="Обработка..." if busy else "Обработать")
        self.youtube_btn.config(state=state if self.result is not None else "disabled")
        if not busy and self.result is None:
            self.open_folder_btn.config(state="disabled")
            self.publish_btn.config(state="disabled")
        if status:
            self.status_var.set(status)

    def _log(self, message: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", str(message) + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")


def main() -> int:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except tk.TclError:
        pass

    PipelineGUI(root).root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())