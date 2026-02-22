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

## Reward System

OpenAI Five 스타일 리워드 파이프라인:

| Reward | Value | Description |
|--------|-------|-------------|
| Kill (enemy) | +3.0 | 적 서번트 처치 |
| Death | -1.0 | 사망 |
| Creep Kill | +0.5 | 크립 처치 (레벨 12 미만만) |
| Level Up | +0.5 | 레벨업 |
| Damage Ratio | +3.0 | 적 maxHP 100% 기여 시 (팀 분배) |
| Heal Ratio | +1.0 | 자신 HP 회복 비율 |
| Score Point | +2.0 | 팀 점수 획득 |
| Portal Use | +0.05 | 포탈 사용 (포만도 감소: 0.995^n) |
| Friendly Kill | -3.0 | 아군 처치 |
| Idle Penalty | -0.003 | 이동/전투 없이 정지 |
| Skill Points Held | -0.02 | 미사용 스킬 포인트 보유 |
| Win / Lose / Timeout | +10 / -5 / -2 | 게임 종료 |

**후처리**: Team Spirit (τ=0.5) → Zero-Sum → Time Decay (0.7^(t/600))

**자동 커리큘럼**:
- 포탈 리워드: 사용할수록 감소 (`0.05 × 0.995^count`), ~500회 후 자동 소멸
- 크립 리워드: 레벨 12 이상에서 0 (PvP 전환 유도)

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

## Credits & Acknowledgements

### Fate/Another

이 프로젝트는 **Fate/Another** 커스텀 맵 위에서 동작합니다. Fate/Another는 수년간 수많은 플레이어들에게 사랑받아온 명작 AOS 맵으로, 원작의 서번트들을 워크래프트 3 엔진 위에 놀라운 완성도로 구현해낸 작품입니다. 맵 제작자분들의 헌신과 창의력에 깊은 존경을 표합니다. 본 프로젝트는 원본 맵의 게임플레이를 AI 학습 환경으로 활용하기 위한 연구 목적의 프로젝트이며, 원작의 저작권을 침해할 의도가 없습니다.

### Type-Moon

**Fate/stay night**을 비롯한 Fate 시리즈의 세계관, 캐릭터, 설정은 모두 [Type-Moon](https://www.typemoon.com/)의 창작물입니다. 세이버, 아처, 랜서, 라이더, 캐스터, 어새신, 버서커 - 이 매력적인 서번트들이 없었다면 Fate/Another도, 이 프로젝트도 존재하지 않았을 것입니다. Nasu Kinoko와 Takeuchi Takashi를 비롯한 Type-Moon 여러분께 경의를 표합니다.

### Blizzard Entertainment & Warcraft III

**Warcraft III: Reign of Chaos / The Frozen Throne**는 [Blizzard Entertainment](https://www.blizzard.com/)가 만든 전설적인 RTS입니다. 강력한 World Editor와 JASS 스크립팅, 그리고 커스텀 맵 생태계는 수십 년이 지난 지금까지도 모더와 플레이어들의 창작을 가능하게 하고 있습니다. DotA, Fate/Another, 그리고 이 RL 프로젝트까지 - 모두 워크래프트 3라는 위대한 플랫폼 위에 세워진 것입니다.

### Open Source

- [EasyHook](https://github.com/EasyHook/EasyHook) - DLL injection framework
- [Cirnix.JassNative](https://github.com/ScorpioN315/Cirnix.JassNative) - WC3 JASS native runtime
- [StormLib](https://github.com/ladislav-zezula/StormLib) - MPQ archive library

