# FateAnotherRL

Warcraft III 커스텀 맵 **Fate/Another FS 2.6A**를 PPO 강화학습으로 학습시키는 프로젝트.

12인 FFA (6v6) 환경에서 각 영웅의 스킬, 아이템, 이동, 스탯 분배를 AI가 학습합니다.

## Architecture

```
WC3 Game (JASS patched)
    ↓ UDP (state every 0.1s)
C++ Inference Server (TorchScript)
    ↓ action → WC3 / rollout → .pt files
Python Trainer (PPO offline)
    ↓ model_latest.pt (hot-reload)
C++ Inference Server ↺
```

| Layer | Language | Directory | Role |
|-------|----------|-----------|------|
| Game Plugin | C# | `rl_comm/` | UDP로 게임 상태 전송, AI 행동 수신 |
| Map Patch | Python | `MapPatch/` | JASS 코드 패치 (RL 환경 구성) |
| Inference | C++ | `inference_server/` | TorchScript 추론 + 롤아웃 수집 |
| Training | Python | `fateanother_rl/` | PPO 학습 + 모델 export |
| DLL Injection | C# | `JassNative/`, `JNLoader/`, `EasyHook/` | WC3에 커스텀 native 주입 |
| Build | Python | `assemble.py` | MSBuild 통합 빌드 |

## Model

- **Policy**: Shared LSTM (256-dim) - 12 영웅 공용 (hero_id one-hot)
- **Action Space**: 11 discrete heads (스킬, 아이템, 스탯 등) + 2 continuous (이동, 포인팅)
- **Training**: Offline PPO (C++에서 롤아웃 파일 수집 → Python에서 배치 학습)
- **Hot-Reload**: 학습 완료된 모델을 C++ 서버가 자동 감지하여 교체

## Quick Start

### Windows (로컬 학습)

```bat
run_windows.bat
```

### Linux

```bash
./run_linux.sh
```

### Docker (분산 학습)

```bash
./run_docker.sh
```

15 WC3 클라이언트 + 5 Inference 서버 + 1 Trainer + TensorBoard

## Build

```bash
python assemble.py
```

JassNative → JNLoader → RLCommPlugin → War3Client 순서로 빌드.

## Map Patch

```bash
cd MapPatch
python rl_patch.py
```

원본 맵(`fateanother_verFS26A_original.w3x`)을 RL 환경용으로 패치:
- 12인 자동 영웅 선택 (팀 고정 배치)
- 어빌리티 base type 변환 (IssueOrder 호환)
- AntiMapHack 제거, 크래시 버그 수정
- RL 이벤트 훅 (KILL, LVUP, ALARM, DONE 등)
- JassNative TCP 통신 native 선언

## Project Structure

```
FateAnotherRL/
├── fateanother_rl/        # Python RL framework
│   ├── model/             #   policy, encoder, action masking
│   ├── training/          #   PPO, buffer, reward, trainer
│   ├── data/              #   hero/item constants
│   ├── env/               #   state parser, env manager
│   ├── selfplay/          #   checkpoint pool
│   └── scripts/           #   train.py, init_models.py
├── inference_server/      # C++ inference + rollout writer
│   ├── include/           #   headers
│   └── src/               #   implementation
├── rl_comm/               # C# WC3 plugin (UDP state/action)
├── MapPatch/              # JASS map patcher
├── JassNative/            # WC3 custom native runtime
├── JNLoader/              # JassNative loader
├── EasyHook/              # DLL injection framework
├── docker/                # Docker distributed training
├── thirdparty/            # Pre-built DLLs
├── assemble.py            # Build orchestrator
├── run_windows.bat        # Windows launcher
├── run_linux.sh           # Linux launcher
└── run_docker.sh          # Docker launcher
```

## Prerequisites

### Warcraft III 1.28.5.7680 (필수)

이 프로젝트는 **Warcraft III 1.28.5.7680** 버전에서만 동작합니다. JassNative DLL 인젝션과 메모리 오프셋이 이 버전에 맞춰져 있으므로, 다른 버전에서는 실행되지 않습니다.

다운로드: https://drive.google.com/file/d/1Kuo-8hss8fX-IxLAdX62BFrpZdk9wPYI/view?usp=sharing

> `War3Client/` 디렉토리에 압축 해제 후 사용하세요.

