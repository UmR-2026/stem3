# Third-party notices — stem3 binary distribution

`stem3.exe` is a single-file bundle. It embeds third-party libraries and two
pre-trained model checkpoints. Each component keeps its own license; the MIT
license in `LICENSE` covers **only the stem3 source code in this repository**.

This file is attached to every Release. Keep it alongside `stem3.exe` if you
redistribute the binary.

## 1. Model weights (embedded in the exe)

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `UVR_MDXNET_Main.onnx` | 66,759,516 | `900fb63c1a2405fe860316ca03c831ce4c7a4129ff1fbc55f99aee4422b98592` |
| `UVR_MDXNET_KARA_2.onnx` | 52,787,030 | `c05b3b7a3ab73362c85511dd0386ad173f28570e23ce1d8d3f87391396a85c08` |

Origin: **Ultimate Vocal Remover (UVR)** — Anjok07 (DilanBoskan), KimberleyJSN
and the UVR development community — <https://github.com/Anjok07/ultimatevocalremovergui>.
Distributed via <https://github.com/TRvlvr/model_repo>; the ONNX exports with
metadata used here come from the `source-separation-models` release of
<https://github.com/k2-fsa/sherpa-onnx>.

License: **MIT, with an attribution request**. Upstream asks that applications
using UVR-trained models credit UVR and its developers — this notice serves as
that credit.

Model architecture: KUIELab-MDX-Net (Woosung Choi, Minseok Kim, Jaehwa Chung,
Daewon Lee, Soonyoung Jung), *KUIELab-MDX-Net: A Two-Stream Neural Network for
Music Demixing*, arXiv:2111.12203.

```
MIT License

Copyright (c) Ultimate Vocal Remover project authors
(Anjok07 / DilanBoskan, KimberleyJSN, and the UVR development community)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Qt / PySide6 (DLLs embedded in the exe)

Qt for Python (PySide6) is dual-licensed under **LGPLv3 / GPLv3**. This build
uses Qt under the LGPLv3 terms, via dynamically loaded DLLs (no static linking).

- LGPLv3 text: <https://www.gnu.org/licenses/lgpl-3.0.html>
- Qt licensing overview: <https://doc.qt.io/qt-6/licensing.html>

**Relinking (LGPLv3 section 4):** You may modify the application and re-link it
against a different version of the Qt libraries. This repository ships the
complete source, `requirements.txt` and `build.bat`; running `build.bat`
rebuilds `dist\stem3.exe` from source against whatever PySide6 version is
installed. No object code or source needed for that purpose is withheld.

## 3. Other embedded components

| Component | License | Upstream |
| --- | --- | --- |
| sherpa-onnx | Apache-2.0 | <https://github.com/k2-fsa/sherpa-onnx> |
| ONNX Runtime (DLL shipped with sherpa-onnx) | MIT | <https://github.com/microsoft/onnxruntime> |
| libsndfile | LGPL-2.1-or-later | <https://github.com/libsndfile/libsndfile> |
| soundfile (Python binding) | BSD-3-Clause | <https://github.com/bastibe/python-soundfile> |
| numpy | BSD-3-Clause | <https://numpy.org/> |
| PyInstaller bootloader | GPLv2 with bootloader exception | <https://github.com/pyinstaller/pyinstaller> |
| Python 3.12 runtime | PSF License | <https://docs.python.org/3/license.html> |

Full license texts are available at the upstream links and in the installed
packages (`<package>-<version>.dist-info/LICENSE*`).

## 4. Scope

The stem3 authors claim no ownership of any third-party component or model
weight listed above. Nothing here grants rights beyond those already granted by
each component's own license.
