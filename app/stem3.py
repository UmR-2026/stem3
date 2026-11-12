# -*- coding: utf-8 -*-
"""stem3 - 原曲分离为 两轨 或 三轨。

两轨模式（mode=2）：UVR_MDXNET_Main 单级推理
    人声(含主唱与和声) / 伴奏
三轨模式（mode=3）：两阶段残差级联
    1) UVR_MDXNET_KARA_2 : 原曲 -> 卡拉OK轨(伴奏+和声) + 主唱人声(残差)
    2) UVR_MDXNET_Main   : 卡拉OK轨 -> 和声伴唱 + 纯伴奏(残差)

默认按 10 秒分段推理，段间 1.5 秒重叠交叉淡化。进度条显示
「已完成段数 / 总段数」——真实进度，不是估算动画。
各轨共用完全相同的分段与淡化权重，故相加仍严格等于原曲。
--whole 可切回整曲一次性推理（无分段边界，但进度不可细分）。

GUI 用 PySide6（Qt 6）：原生 dragEnterEvent/dropEvent 处理拖入，
不再使用自写的 ctypes WNDPROC 子类化。拖到 exe 图标上同样可用。

注意：主唱/和声伴唱的区分是音源分离领域的公认难题，开源模型的
和声轨偏保守。独唱歌曲该轨接近静音属正常，不是故障。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

VERSION = "2.2"

STEMS_BY_MODE = {
    2: ("人声", "伴奏"),
    3: ("人声", "和声", "伴奏"),
}

# libsndfile 1.2.2 实测可用格式（sf.available_formats()），只列常见封装
SUPPORTED_EXTS = {
    ".wav", ".wave", ".mp3", ".flac", ".ogg", ".oga",
    ".aiff", ".aif", ".caf", ".au", ".w64", ".rf64",
}
FORMAT_OK = "WAV / MP3 / FLAC / OGG / AIFF / CAF"
FORMAT_NO = "M4A / AAC / WMA / WavPack"

# 界面上明确写出的输入音频限制
LIMIT_ROWS = (
    (f"支持：{FORMAT_OK}（libsndfile 1.2.2）", "good"),
    (f"不支持：{FORMAT_NO} —— 请先转成 WAV 或 MP3", "bad"),
    ("输入采样率与位深不限；输出统一 44.1 kHz / 立体声 / PCM16 WAV", "hint"),
    ("单声道自动复制为双声道；多声道只取前两轨", "hint"),
    ("输出「原名_人声 / 和声 / 伴奏.wav」，同名文件会被覆盖", "hint"),
    ("整曲载入内存，长音频占用内存较多；单轨峰值 > 1.0 会限幅", "hint"),
    ("和声轨只含真实存在的和声伴唱；独唱歌曲该轨接近静音属正常", "hint"),
)

CHUNK_SEC = 10.0
OVERLAP_SEC = 1.5


def res_root() -> Path:
    """PyInstaller onefile 把资源解压到 sys._MEIPASS。"""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


MODEL_MAIN = res_root() / "models" / "UVR_MDXNET_Main.onnx"
MODEL_KARA = res_root() / "models" / "UVR_MDXNET_KARA_2.onnx"


def check_assets():
    missing = [p.name for p in (MODEL_MAIN, MODEL_KARA) if not p.is_file()]
    if missing:
        raise RuntimeError("内嵌模型缺失：" + ", ".join(missing))


def load_audio(path: Path):
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(
            f"无法解码 {path.name}：{exc}。支持 {FORMAT_OK}；{FORMAT_NO} 请先转码。"
        ) from exc
    if data.shape[1] == 1:
        # 单声道复制成双声道，MDX 网络按 4 通道输入（L/R 的实部虚部）设计
        padded = np.zeros((data.shape[0], 2), dtype=np.float32)
        padded[:, 0] = data[:, 0]
        padded[:, 1] = data[:, 0]
        data = padded
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data.T), int(sr)


def build_separator(model: Path, threads: int):
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(model)),
            num_threads=threads,
            debug=False,
            provider="cpu",
        )
    )
    if not cfg.validate():
        raise RuntimeError("模型配置校验失败：" + model.name)
    return sherpa_onnx.OfflineSourceSeparation(cfg)


def _make_blocks(length: int, sr: int):
    """切出 (start, end) 列表，相邻块重叠 OVERLAP_SEC 秒。"""
    step = max(1, int(CHUNK_SEC * sr))
    ov = max(1, int(OVERLAP_SEC * sr))
    if step <= 2 * ov:
        raise ValueError("分段参数不合理")
    blocks = []
    s = 0
    while s < length:
        e = min(length, s + step)
        if length - e < ov:  # 尾巴太短就并进当前块
            e = length
        blocks.append((s, e))
        if e >= length:
            break
        s = e - ov
    return blocks, ov


def _run_stage(seg, seg_sr, mode, sp_kara, sp_main, log=None, idx=0, total=1):
    """对一段音频跑 1 或 2 级模型，返回 (输出采样率, stem 列表)。"""
    import numpy as np

    if mode == 2:
        out = sp_main.process(sample_rate=seg_sr, samples=np.ascontiguousarray(seg))
        stems = [np.asarray(out.stems[0].data, dtype=np.float32),
                 np.asarray(out.stems[1].data, dtype=np.float32)]
        out_sr = int(out.sample_rate)
        del out
        return out_sr, stems

    out = sp_kara.process(sample_rate=seg_sr, samples=np.ascontiguousarray(seg))
    # KARA_2 的预测目标是「去掉主唱后剩下的」，主唱在残差里
    karaoke = np.asarray(out.stems[0].data, dtype=np.float32)
    lead = np.asarray(out.stems[1].data, dtype=np.float32)
    out_sr = int(out.sample_rate)
    del out
    if log is not None:
        log(("progress", idx, total, f"第 {idx + 1}/{total} 段 · ② 和声与伴奏"))
    out = sp_main.process(sample_rate=out_sr, samples=np.ascontiguousarray(karaoke))
    # Main 的预测目标是人声；此时输入已无主唱，故为和声伴唱
    harmony = np.asarray(out.stems[0].data, dtype=np.float32)
    instrumental = np.asarray(out.stems[1].data, dtype=np.float32)
    del out, karaoke
    return out_sr, [lead, harmony, instrumental]


def separate(x, sr: int, threads: int, log, whole: bool = False, mode: int = 3):
    """返回 (stems_tuple, 输出采样率)。stems 数量由 mode 决定。

    log 收到 ("progress", done, total, stage) 元组时表示真实进度：
    done 是已完成的分段数，不是计时估算。
    """
    import numpy as np

    check_assets()
    t_start = time.time()
    k = len(STEMS_BY_MODE[mode])
    sp_kara = build_separator(MODEL_KARA, threads) if mode == 3 else None
    sp_main = build_separator(MODEL_MAIN, threads)
    log(f"   模式：{'两轨 人声/伴奏（单级）' if mode == 2 else '三轨 人声/和声/伴奏（两级）'}")

    if whole:
        log(("progress", 0, 1, "整曲推理（不分段，进度不可细分）"))
        out_sr, stems = _run_stage(x, sr, mode, sp_kara, sp_main)
        log(("progress", 1, 1, "完成"))
        log(f"   推理完成，用时 {time.time() - t_start:.0f}s")
        n = min(s.shape[1] for s in stems)
        return tuple(s[:, :n] for s in stems), out_sr

    length = x.shape[1]
    blocks, ov = _make_blocks(length, sr)
    total = len(blocks)
    log(f"   分 {total} 段推理（每段约 {CHUNK_SEC:.0f}s，重叠 {OVERLAP_SEC:.1f}s 交叉淡化）")

    acc = None
    wsum = None
    out_sr = None
    length_out = 0
    ov_out = 0

    for idx, (s, e) in enumerate(blocks):
        seg = np.ascontiguousarray(x[:, s:e])
        stage = (f"第 {idx + 1}/{total} 段 · ① 主唱提取" if mode == 3
                 else f"第 {idx + 1}/{total} 段 · 人声/伴奏分离")
        log(("progress", idx, total, stage))
        this_sr, stems = _run_stage(seg, sr, mode, sp_kara, sp_main, log, idx, total)
        if out_sr is None:
            out_sr = this_sr
            length_out = max(1, int(round(length * out_sr / sr)))
            ov_out = max(1, int(round(ov * out_sr / sr)))
            acc = [np.zeros((2, length_out), dtype=np.float32) for _ in range(k)]
            wsum = np.zeros(length_out, dtype=np.float32)
        del seg

        m = min(a.shape[1] for a in stems)
        start = max(0, min(int(round(s * out_sr / sr)), length_out - 1))
        m = min(m, length_out - start)
        if m > 0:
            # 相邻块重叠区权重互补（0→1 与 1→0），各轨共用同一权重，
            # 因此 Σ权重 = 1，各轨相加仍严格等于原曲。
            w = np.ones(m, dtype=np.float32)
            if ov_out < m:
                if idx > 0:
                    w[:ov_out] = np.linspace(0.0, 1.0, ov_out, dtype=np.float32)
                if idx < total - 1:
                    w[-ov_out:] = np.linspace(1.0, 0.0, ov_out, dtype=np.float32)
            for j, arr in enumerate(stems):
                acc[j][:, start:start + m] += arr[:, :m] * w[None, :]
            wsum[start:start + m] += w
        del stems
        log(("progress", idx + 1, total, f"第 {idx + 1}/{total} 段完成"))

    wsum = np.maximum(wsum, 1e-6)
    for a in acc:
        a /= wsum[None, :]
    log(f"   推理完成，用时 {time.time() - t_start:.0f}s")
    return tuple(acc), out_sr


def write_stems(stems, names, sr: int, src: Path, outdir: Path, log):
    import numpy as np
    import soundfile as sf

    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    peak = 0.0
    for arr, name in zip(stems, names):
        y = np.ascontiguousarray(arr.T)
        peak = max(peak, float(np.max(np.abs(y))) if y.size else 0.0)
        np.clip(y, -1.0, 1.0, out=y)
        dst = outdir / f"{src.stem}_{name}.wav"
        sf.write(str(dst), y, samplerate=sr, subtype="PCM_16")
        written.append(dst)
        log(f"   ✓ {dst.name}  ({y.shape[0] / sr:.1f}s)")
    if peak > 1.0:
        log(f"   注意：峰值 {peak:.2f} 已限幅到 1.0")
    return written


def process_one(path: Path, args, log):
    import gc

    log(f"\n▸ {path.name}")
    if not path.is_file():
        raise FileNotFoundError("找不到文件：" + str(path))
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise RuntimeError(
            f"不支持 {path.suffix.lower() or '无扩展名'}：只支持 {FORMAT_OK}；"
            f"{FORMAT_NO} 请先转成 WAV 或 MP3。"
        )

    x, sr = load_audio(path)
    log(f"   输入 {x.shape[1] / sr:.1f}s / {sr} Hz / {x.shape[0]} 声道")
    stems, out_sr = separate(x, sr, args.threads, log, whole=args.whole, mode=args.mode)
    del x
    gc.collect()

    outdir = Path(args.outdir).resolve() if args.outdir else path.parent
    written = write_stems(stems, STEMS_BY_MODE[args.mode], out_sr, path, outdir, log)
    log(f"   输出目录：{outdir}")
    return written


# --------------------------------------------------------------------------
# PySide6 界面
# --------------------------------------------------------------------------

QSS = """
QWidget { color: #1f2937;
          font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; }
QWidget#main, QScrollArea#scroll, QWidget#content { background: #f6f8fb; }
QCheckBox::indicator { width: 16px; height: 16px;
    border: 2px solid #94a3b8; border-radius: 4px; background: #ffffff; }
QCheckBox::indicator:hover { border-color: #60a5fa; }
QCheckBox::indicator:checked { border-color: #2563eb; background: #2563eb; }
QLabel#title { color: #111827; }
QLabel#subtitle { color: #6b7280; }
QFrame#drop { border: 2px dashed #a9c6e4; border-radius: 12px; background: #eef5fc; }
QFrame#drop[active="true"] { border-color: #2563eb; background: #dceafd; }
QLabel#dropMain { color: #1d4ed8; }
QLabel#dropSub { color: #6b8bad; }
QGroupBox { border: 1px solid #e2e8f0; border-radius: 10px; background: #ffffff;
            margin-top: 14px; padding: 11px 12px 8px; color: #334155; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QCheckBox { spacing: 8px; padding: 3px 0; background: transparent; }
QRadioButton { border: 2px solid #e2e8f0; border-radius: 9px;
    background: #ffffff; color: #334155; padding: 9px 12px; spacing: 0; }
QRadioButton:hover { border-color: #93b8de; background: #f7fbff; }
QRadioButton:checked { border-color: #2563eb; background: #eaf2ff; color: #1d4ed8; }
QRadioButton::indicator { width: 0; height: 0; border: none;
    background: transparent; image: none; }
QLineEdit { border: 1px solid #d5dde7; border-radius: 8px; padding: 7px 10px;
            background: #ffffff; selection-background-color: #bfdbfe; }
QLineEdit:focus { border-color: #60a5fa; }
QPushButton { border: 1px solid #d5dde7; border-radius: 8px; padding: 7px 14px;
              background: #ffffff; color: #334155; }
QPushButton:hover { background: #f1f6fc; border-color: #93b8de; }
QPushButton:pressed { background: #e3eefb; }
QPushButton#primary { background: #2563eb; color: #ffffff; border: none; font-weight: 600; padding: 9px 18px; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #a9bde0; }
QProgressBar { border: none; border-radius: 6px; background: #e6ecf4; height: 10px; }
QProgressBar::chunk { border-radius: 6px; background: #2563eb; }
QPlainTextEdit { border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff;
                 font-family: Consolas, "Cascadia Mono", monospace; padding: 8px; }
QLabel#status { color: #1d4ed8; }
QLabel#hint { color: #64748b; }
QLabel#good { color: #15803d; }
QLabel#bad { color: #b91c1c; }
"""


def run_qt_gui(initial_files, args):
    """Qt 界面入口。所有 Qt import 都放在函数内，--nogui 不依赖 Qt。"""
    from html import escape

    from PySide6.QtCore import Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent, QFont
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
        QRadioButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
    )

    class Worker(QThread):
        log_sig = Signal(str)
        progress_sig = Signal(int, int, str)
        finished_sig = Signal(int, int)

        def __init__(self, files, cfg):
            super().__init__()
            self.files = list(files)
            self.cfg = cfg

        def run(self):
            def log(msg):
                if isinstance(msg, tuple) and msg and msg[0] == "progress":
                    self.progress_sig.emit(int(msg[1]), int(msg[2]), str(msg[3]))
                else:
                    self.log_sig.emit(str(msg))

            ok = fail = 0
            for f in self.files:
                try:
                    process_one(Path(f), self.cfg, log)
                    ok += 1
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    self.log_sig.emit(f"   ✗ 失败：{exc}")
                    self.log_sig.emit(traceback.format_exc().strip().splitlines()[-1])
            self.finished_sig.emit(ok, fail)

    class MainWindow(QWidget):
        def __init__(self, cfg, files=None):
            super().__init__()
            self.cfg = cfg
            self.worker = None
            self.running = False
            self.t0 = 0.0
            self.done = 0
            self.total = 1
            self.stage = "待机"
            self.setWindowTitle(f"音轨分离 · stem3  v{VERSION}")
            self.setObjectName("main")
            self.setMinimumSize(600, 540)
            self.resize(680, 790)
            self.setAcceptDrops(True)
            self._build()
            self.setStyleSheet(QSS)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(200)
            if files:
                QTimer.singleShot(200, lambda: self.start_files(files))

        @staticmethod
        def _font(factor, bold=False):
            """字号按应用基准点值缩放（pt 随系统 DPI 自适应），不写死 px。"""
            base = QApplication.instance().font()
            f = QFont(base)
            f.setPointSizeF(max(8.0, base.pointSizeF() * factor))
            f.setBold(bold)
            return f

        def _build(self):
            # 内容放进滚动区：小屏 / 高 DPI 下按自然高度排布，绝不压缩各行文字
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea(self)
            scroll.setObjectName("scroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            content = QWidget()
            content.setObjectName("content")
            root = QVBoxLayout(content)
            root.setContentsMargins(22, 16, 22, 12)
            root.setSpacing(7)

            title = QLabel("音轨分离")
            title.setObjectName("title")
            title.setFont(self._font(1.7, True))
            subtitle = QLabel("原曲 → 两轨 或 三轨 · 本地离线推理，无需联网")
            subtitle.setObjectName("subtitle")
            subtitle.setFont(self._font(0.95))
            root.addWidget(title)
            root.addWidget(subtitle)

            self.drop_frame = QFrame()
            self.drop_frame.setObjectName("drop")
            self.drop_frame.setProperty("active", "false")
            self.drop_frame.setMinimumHeight(68)
            dv = QVBoxLayout(self.drop_frame)
            dv.setContentsMargins(12, 8, 12, 8)
            dv.setSpacing(3)
            main = QLabel("把音频文件拖到这里")
            main.setObjectName("dropMain")
            main.setFont(self._font(1.2, True))
            main.setAlignment(Qt.AlignCenter)
            sub = QLabel("也可拖到程序图标上，或点下方「选择音频文件」")
            sub.setObjectName("dropSub")
            sub.setFont(self._font(0.9))
            sub.setAlignment(Qt.AlignCenter)
            dv.addWidget(main)
            dv.addWidget(sub)
            root.addWidget(self.drop_frame)

            mode_box = QGroupBox("分离模式")
            mv = QVBoxLayout(mode_box)
            mv.setSpacing(6)
            self.rb2 = QRadioButton("两轨　人声（主唱＋和声） / 伴奏　·　单级模型，耗时更短")
            self.rb3 = QRadioButton("三轨　人声（主唱） / 和声（伴唱） / 伴奏　·　两级残差级联")
            (self.rb3 if self.cfg.mode == 3 else self.rb2).setChecked(True)
            mv.addWidget(self.rb2)
            mv.addWidget(self.rb3)
            root.addWidget(mode_box)

            out_box = QGroupBox("导出位置")
            ov = QVBoxLayout(out_box)
            row = QHBoxLayout()
            self.out_edit = QLineEdit(self.cfg.outdir)
            self.out_edit.setPlaceholderText("留空 = 保存到原曲所在目录")
            browse = QPushButton("浏览…")
            browse.clicked.connect(self._browse_out)
            row.addWidget(self.out_edit, 1)
            row.addWidget(browse)
            ov.addLayout(row)
            ohint = QLabel("留空 = 原曲所在目录；批量处理时统一导出到这里")
            ohint.setObjectName("hint")
            ohint.setFont(self._font(0.9))
            ov.addWidget(ohint)
            root.addWidget(out_box)

            # 单个富文本标签承载全部限制：QLabel 能正确算出整块多行文本的高度，
            # 不会像多个 wordWrap 标签那样被布局压成一行导致文字重叠。
            lim_box = QGroupBox("输入音频限制")
            lv = QVBoxLayout(lim_box)
            colors = {"good": "#15803d", "bad": "#b91c1c", "hint": "#64748b"}
            html = "".join(
                f'<div style="margin:2px 0;"><font color="{colors[kind]}">{escape(text)}</font></div>'
                for text, kind in LIMIT_ROWS
            )
            limits = QLabel(html)
            limits.setTextFormat(Qt.RichText)
            limits.setWordWrap(True)
            limits.setFont(self._font(0.94))
            limits.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            lv.addWidget(limits)
            root.addWidget(lim_box)

            self.whole_cb = QCheckBox("整曲推理（不分段，进度不可细分）")
            self.whole_cb.setChecked(self.cfg.whole)
            root.addWidget(self.whole_cb)

            self.status = QLabel("待机")
            self.status.setObjectName("status")
            self.status.setFont(self._font(1.0, True))
            root.addWidget(self.status)
            self.bar = QProgressBar()
            self.bar.setRange(0, 1)
            self.bar.setValue(0)
            self.bar.setTextVisible(False)
            root.addWidget(self.bar)

            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMinimumHeight(66)
            self.log.setMaximumHeight(120)
            self.log.setFont(self._font(0.9))
            root.addWidget(self.log)
            root.addStretch(1)

            scroll.setWidget(content)
            outer.addWidget(scroll, 1)

            # 按钮固定在滚动区外，小屏下也始终可见
            brow = QHBoxLayout()
            brow.setContentsMargins(22, 4, 22, 14)
            self.pick_btn = QPushButton("选择音频文件…")
            self.pick_btn.setObjectName("primary")
            self.pick_btn.clicked.connect(self._choose)
            brow.addWidget(self.pick_btn)
            brow.addStretch(1)
            outer.addLayout(brow)

            self.log.appendPlainText(
                f"支持 {FORMAT_OK}；{FORMAT_NO} 需先转码。进度按已完成分段显示。"
            )

        def _browse_out(self):
            d = QFileDialog.getExistingDirectory(self, "选择导出目录", self.out_edit.text())
            if d:
                self.out_edit.setText(d)

        def _choose(self):
            exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择原曲（可多选）", "", f"音频文件 ({exts});;所有文件 (*.*)"
            )
            if files:
                self.start_files(files)

        def _set_drop_active(self, active):
            self.drop_frame.setProperty("active", "true" if active else "false")
            self.drop_frame.style().unpolish(self.drop_frame)
            self.drop_frame.style().polish(self.drop_frame)

        def dragEnterEvent(self, event: QDragEnterEvent):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                self._set_drop_active(True)

        def dragMoveEvent(self, event: QDragMoveEvent):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dragLeaveEvent(self, event: QDragLeaveEvent):
            self._set_drop_active(False)
            event.accept()

        def dropEvent(self, event: QDropEvent):
            self._set_drop_active(False)
            if not event.mimeData().hasUrls():
                return
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            files = [f for f in files if f and Path(f).is_file()]
            event.acceptProposedAction()
            if files:
                self.start_files(files)

        def start_files(self, files):
            if self.running:
                self.log.appendPlainText("   （正在处理中，忽略本次操作）")
                return
            files = [f for f in files if Path(f).is_file()]
            if not files:
                return
            self.cfg.mode = 3 if self.rb3.isChecked() else 2
            self.cfg.outdir = self.out_edit.text().strip()
            self.cfg.whole = self.whole_cb.isChecked()
            self.running = True
            self.done, self.total, self.stage = 0, 1, "准备中"
            self.t0 = time.time()
            self.bar.setRange(0, 1)
            self.bar.setValue(0)
            self.pick_btn.setEnabled(False)
            self.log.clear()
            self.log.appendPlainText(f"已加入 {len(files)} 个文件")
            self.worker = Worker(files, self.cfg)
            self.worker.log_sig.connect(self.log.appendPlainText)
            self.worker.progress_sig.connect(self._on_progress)
            self.worker.finished_sig.connect(self._on_finished)
            self.worker.start()

        def _on_progress(self, done, total, stage):
            self.done, self.total, self.stage = done, max(1, total), stage
            self.bar.setRange(0, max(1, total))
            self.bar.setValue(done)

        def _on_finished(self, ok, fail):
            self.running = False
            self.bar.setValue(self.bar.maximum())
            self.pick_btn.setEnabled(True)
            self.status.setText(f"完成：成功 {ok} 个，失败 {fail} 个")

        def _tick(self):
            if not self.running:
                return
            if self.total > 1:
                detail = f"{self.done}/{self.total} 段 ({int(100 * self.done / self.total)}%)"
            else:
                detail = "整曲推理"
            elapsed = int(time.time() - self.t0)
            self.status.setText(
                f"{self.stage}  ·  {detail}  ·  已用 {elapsed // 60:02d}:{elapsed % 60:02d}"
            )

    app = QApplication.instance() or QApplication(sys.argv)
    base = app.font()
    if base.pointSizeF() < 9.0:  # 只设下限；实际点值交给系统 DPI / 用户字体设置
        base.setPointSizeF(9.0)
    app.setFont(base)
    win = MainWindow(args, initial_files)
    win.show()
    return app.exec()


def run_cli(files, args):
    last_done = [-1]

    def log(msg):
        if isinstance(msg, tuple) and msg and msg[0] == "progress":
            _, done, total, stage = msg
            if done != last_done[0]:
                if total > 1:
                    print(f"   进度 {done}/{total} 段  ({stage})", flush=True)
                else:
                    print(f"   {stage}", flush=True)
                last_done[0] = done
        else:
            print(msg, flush=True)

    if not files:
        log("用法：stem3.exe <音频文件> [更多文件…]  [--mode 2|3] [--outdir 目录] "
            "[--threads N] [--nogui] [--whole]")
        log(f"支持 {FORMAT_OK}；{FORMAT_NO} 需先转码。")
        return 2
    ok = 0
    for f in files:
        try:
            process_one(Path(f), args, log)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            log(f"   ✗ 失败：{exc}")
            traceback.print_exc()
    log(f"\n完成：{ok}/{len(files)}")
    return 0 if ok == len(files) else 1


def _install_crash_log():
    """windowed 模式没有 stderr，把未捕获异常落盘，便于排查闪退。"""
    def hook(exc_type, exc, tb):
        try:
            with open(Path.cwd() / "stem3_crash.log", "a", encoding="utf-8") as fh:
                fh.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                traceback.print_exception(exc_type, exc, tb, file=fh)
        except OSError:
            pass
        try:
            if sys.__excepthook__:
                sys.__excepthook__(exc_type, exc, tb)
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


def main():
    parser = argparse.ArgumentParser(add_help=True, description="原曲分离 → 两轨 / 三轨")
    parser.add_argument("files", nargs="*", help="音频文件（可多个）")
    parser.add_argument("--mode", type=int, choices=(2, 3), default=3,
                        help="2=人声/伴奏，3=人声/和声/伴奏（默认 3）")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4), help="推理线程数")
    parser.add_argument("--outdir", default="", help="导出目录，留空=原曲所在目录")
    parser.add_argument("--nogui", action="store_true", help="不弹界面，纯控制台输出")
    parser.add_argument("--whole", action="store_true", help="整曲一次性推理（无分段边界，但进度不可细分）")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()
    _install_crash_log()

    # PyInstaller --windowed 模式不分配控制台，sys.stdout/stderr 为 None，
    # 需要兜底，否则 print / traceback 会直接抛异常。
    if sys.stdout is None:
        import io

        sink = io.StringIO()
        if args.nogui:
            try:
                sink = open(Path.cwd() / "stem3.log", "a", encoding="utf-8")
            except OSError:
                pass
        sys.stdout = sys.stderr = sink

    if args.nogui:
        return run_cli(args.files, args)
    try:
        return run_qt_gui(args.files, args)
    except Exception:  # Qt 不可用时退回控制台
        traceback.print_exc()
        return run_cli(args.files, args)


if __name__ == "__main__":
    sys.exit(main())
