# stem3

把一首原曲拆成 **人声 / 和声伴唱 / 伴奏**（三轨），或 **人声 / 伴奏**（两轨）的本地离线工具。

单文件 Windows exe，零外部依赖：Python 3.12、Qt 6、onnxruntime、两个 UVR ONNX 模型全部内嵌，不需要装 CUDA / Python / ffmpeg，音频不出本机。

## 特性

- **两轨 / 三轨** 切换；窗口内拖入、拖到 exe 图标、命令行批量
- **真实进度**：按已完成分段数显示（`3/8 段 (38%)`），不是估算动画
- **能量守恒**：各轨相加严格等于原曲（残差法 + 交叉淡化）
- 导出位置可设，留空输出到原曲同目录
- 全程 CPU 推理，无需联网

## 使用

按下方「从源码构建」得到 `dist\stem3.exe`，双击打开：

| 方式 | 操作 |
| --- | --- |
| 拖入 | 把音频拖到窗口中的虚线框 |
| 拖到图标 | 把音频拖到 `stem3.exe` 图标上 |
| 按钮 | 点「选择音频文件…」，可多选 |
| 命令行 | `stem3.exe song.mp3 --nogui --mode 2 --outdir D:\out` |

命令行参数：

```
--mode 2|3     2 = 人声/伴奏，3 = 人声/和声/伴奏（默认 3）
--outdir DIR   导出目录，留空 = 原曲所在目录
--threads N    推理线程数（默认 min(8, CPU 核数)）
--whole        整曲一次性推理（无分段边界，但进度不可细分）
--nogui        纯控制台，不弹窗口
```

输出：`原名_人声.wav` / `原名_和声.wav` / `原名_伴奏.wav`，44.1 kHz 立体声 PCM16。

## 输入音频限制

| | |
| --- | --- |
| 支持 | WAV / MP3 / FLAC / OGG / AIFF / CAF（libsndfile 1.2.2） |
| 不支持 | M4A / AAC / WMA / WavPack —— 请先转成 WAV 或 MP3 |
| 采样率 / 位深 | 不限；输出统一 44.1 kHz / 立体声 / PCM16 |
| 声道 | 单声道自动复制为双声道；多声道只取前两轨 |
| 其他 | 整曲载入内存，长音频占用内存较多；单轨峰值 > 1.0 会限幅；同名输出会被覆盖 |

## 原理

两级残差级联。sherpa-onnx 的 UVR 分离里 `stems[0]` 是模型预测目标，`stems[1]` 恒为 `输入 − stems[0]`：

1. `UVR_MDXNET_KARA_2` 的预测目标是「去掉主唱后剩下的」，于是
   `原曲 → 卡拉OK轨(伴奏 + 和声) + 主唱`
2. 把卡拉OK轨喂给 `UVR_MDXNET_Main`（预测目标为人声），得到
   `卡拉OK轨 → 和声伴唱 + 纯伴奏`

三轨相加恒等于原曲。默认按 10 秒分段、1.5 秒重叠交叉淡化；各轨共用同一分段与权重，所以守恒仍然成立（实测还原误差 rms 4.9e-5，只是 PCM16 量化噪声）。分段也让进度条能反映真实完成量，而不是播放一个假动画。

两轨模式只跑一次 `UVR_MDXNET_Main`，人声轨含主唱与和声，耗时更短。

## 已知边界

- **和声伴唱分离是音源分离领域的公认难题。** 用合成和声（0.7× 延迟主唱）实测，级联方案只回收约三成（corr +0.273）。独唱歌曲的和声轨接近静音属正常，不是故障。
- 两轨模式单轨峰值可达 1.11，PCM16 逐轨限幅带来 max|err| 0.1075（rms 1.1e-4，约 0.055%）。要严格无削波需改用 32-bit float WAV 输出。
- CPU 推理，26.1 s 音频 / 4 线程：两轨约 25 s，三轨约 39 s。长曲请耐心。

## 从源码构建

要求 Windows x64 + Python 3.12。仓库**不含模型权重，也不含 exe**（exe 约 177 MB，超过 GitHub 单文件 100 MB 限制）。

```bat
git clone https://github.com/UmR-2026/stem3.git
cd stem3
build.bat
```

`build.bat` 会安装依赖、下载并校验模型、用 PyInstaller 打包，产物在 `dist\stem3.exe`。

手动步骤：

```bat
python -m pip install -r requirements.txt
python scripts\fetch_models.py
python -m PyInstaller --noconfirm --clean --onefile --windowed --noupx --name stem3 ^
  --add-data "models\UVR_MDXNET_Main.onnx;models" ^
  --add-data "models\UVR_MDXNET_KARA_2.onnx;models" ^
  --collect-all sherpa_onnx --collect-all soundfile ^
  app\stem3.py
```

只走命令行（`--nogui`）时不需要 PySide6：`python app\stem3.py song.wav --nogui`。

测试：

```bat
python tests\test_layout.py
python tests\test_drop.py
```

## 模型权重与第三方许可

**本仓库不含模型权重**（约 115 MB）。`scripts/fetch_models.py` 在构建时下载并校验 SHA-256：

| 文件 | 大小 | SHA-256 |
| --- | --- | --- |
| `UVR_MDXNET_Main.onnx` | 66,759,516 B | `900fb63c1a2405fe860316ca03c831ce4c7a4129ff1fbc55f99aee4422b98592` |
| `UVR_MDXNET_KARA_2.onnx` | 52,787,030 B | `c05b3b7a3ab73362c85511dd0386ad173f28570e23ce1d8d3f87391396a85c08` |

下载源为 [k2-fsa/sherpa-onnx 的 `source-separation-models` release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models)（给原始 ONNX 加了元数据）。模型原始出处是 [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)（Anjok07 / KimberleyJSN / UVR 社区），经 [TRvlvr/model_repo](https://github.com/TRvlvr/model_repo) 分发，采用 **MIT + 署名要求**：使用这些模型请注明 UVR 项目及其开发者。UVR 项目内其他模型的许可可能不同，使用前请查阅对应模型卡。

主要依赖许可：

| 依赖 | 许可 |
| --- | --- |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Apache-2.0 |
| [PySide6 / Qt](https://doc.qt.io/qt-6/licensing.html) | LGPLv3 / GPLv3（二进制分发请遵守相应条款） |
| [soundfile / libsndfile](https://github.com/libsndfile/libsndfile) | LGPL-2.1+ |
| numpy | BSD-3-Clause |
| onnxruntime（由 sherpa-onnx 附带） | MIT |

## 许可

本仓库源码采用 [MIT](LICENSE)。该许可只覆盖本仓库代码，不覆盖上述第三方依赖与模型权重，也不代表对模型权重的再授权。
