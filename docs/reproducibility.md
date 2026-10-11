# 실행과 결과 해석

## 원본과 정리본의 관계

이 프로젝트는 2025학년도 2학기 자동차인공지능 수업에서 진행한 **경규진 · 조민규 팀 프로젝트**입니다. `archive/submission/`은 최종 제출 코드이고, `src/carla_seg/`는 2026년 10월 포트폴리오 정리 과정에서 실행 방법과 검증을 보완한 코드입니다. 정리 과정의 추가 기능을 2025년 실험 당시 구현 성과로 해석하지 않습니다.

원본 A4_2 체크포인트는 23,381,901개 학습 파라미터와 136개 state-dict 항목을 가지며, 보관된 U-Net 구조에 호환됩니다. 수정본 발표의 A4와 최종 코드의 설정은 일치하지만, 체크포인트의 학습 로그가 없어 발표 수치를 그 체크포인트의 재측정 결과로 단정하지 않습니다.

## 재현을 위해 보완한 부분

| 항목 | 제출 코드 | 정리본 |
| --- | --- | --- |
| 이미지·마스크 대응 | 두 목록을 각각 정렬한 뒤 인덱스로 연결 | 파일명이 같은 RGB/Label만 연결; 불일치는 기본적으로 오류 |
| 누락 파일 | 마지막 샘플 접근 시 실패 가능 | `--skip-missing`을 명시한 경우에만 누락 항목 제외 |
| Dice loss | `classes=13` 전달 | 전체 클래스의 multiclass Dice를 PyTorch로 계산 |
| 출력 클래스 | 최종 계층에 13 하드코딩 | 생성자의 클래스 수 사용; 기본값 13 |
| 지표 단위 | 0–1 비율 뒤에 `%` 표기 | 비율과 백분율을 구분해 기록 |
| 모델 저장 | 마지막 epoch의 가중치만 저장 | 검증 손실 기준 best와 last를 별도로 저장 |
| 실험 기록 | 콘솔 출력 중심 | 설정·seed·epoch 이력·평가 JSON 저장 |
| 실행 환경 | 고정 상대 경로, 시각화 창 필요 | 경로와 장치를 CLI로 지정; 이미지 파일로 결과 저장 |

정리본은 외부 Dice/metric 패키지 대신 PyTorch로 계산합니다. Dice는 batch와 공간 차원을 합산하고 정답에 없는 클래스의 loss를 0으로 처리한 뒤 전체 클래스 수로 평균합니다. IoU는 전체 평가 샘플의 confusion matrix를 누적하고 클래스별 IoU를 계산합니다. 배경(class 0)을 포함한 13개 클래스가 대상이며, 정답과 예측 모두에 없는 클래스는 macro 평균에서 제외하고 클래스별 결과에 `null`로 기록합니다. 이런 설정과 데이터 누락 처리에 따라 다른 구현의 mIoU와 차이가 날 수 있습니다.

원본 DiceLoss의 `classes` 인자는 클래스 인덱스 목록 또는 `None`이어야 합니다. [공식 구현](https://segmentation-models-pytorch.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/dice.html)을 참고했습니다. 원본 실행 환경의 의존성 버전은 남아 있지 않으므로 과거 환경을 완전히 복원한 것으로 주장하지 않습니다.

## 데이터 준비

과제 안내에 기재된 데이터 출처는 [Semantic Segmentation for Self Driving Cars](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge)입니다. 과제에서 제공된 Training/Validation 분할을 사용했으며, 이 저장소는 전체 데이터나 새로운 임의 분할을 제공하지 않습니다. 출처의 이용 조건을 확인하여 데이터를 별도로 준비하세요.

```text
2D Segmentation/
├── Training/
│   ├── RGB/       # 예: 00_00_000.png
│   └── Label/     # RGB와 동일한 파일명
└── Validation/
    ├── RGB/
    └── Label/
```

포트폴리오 정리 시점의 로컬 파일 수는 다음과 같습니다.

| 분할 | RGB | Label | 사용 가능한 쌍 |
| --- | ---: | ---: | ---: |
| Training | 1,000 | 1,000 | 1,000 |
| Validation | 1,000 | 999 | 999 |

`Validation/Label/F9-99.png`가 누락되어 있습니다. `inspect-data`로 먼저 검사하고, 이 스냅샷을 사용할 때에만 `--skip-missing`을 명시합니다. 파일을 보완한 경우에는 옵션을 제거하여 1,000쌍을 검사하세요. 이 누락이 과거 실험 당시에도 존재했는지는 확인할 수 없습니다.

RGB는 600×800(H×W)로 resize한 뒤 ImageNet mean/std로 정규화합니다. 라벨은 red channel의 class ID를 사용하고 nearest-neighbor로 resize합니다. 원본 정책대로 255를 0으로 치환합니다. **색칠된 시각화 마스크를 class-ID 라벨로 입력하면 안 됩니다.** 클래스 정의는 [CARLA 0.8.4의 13개 클래스 문서](https://carla.readthedocs.io/en/0.8.4/cameras_and_sensors/#camera-semantic-segmentation)와 대응합니다. 최신 CARLA 라벨 체계와는 구분해야 합니다.

## 체크포인트

원본 가중치 파일은 `segmentation_model_A4_2.pth`이며 약 89.3 MiB입니다. 일반 Git 기록에 포함하지 않았고 공개 다운로드 링크도 아직 제공하지 않습니다. 보유한 원본 파일을 `weights/`에 두거나 정리본으로 학습하여 생성한 `runs/final/best.pt`를 사용하세요. 외부 가중치 없이도 학습 명령은 실행할 수 있습니다.

원본 checkpoint의 바이트 수와 SHA-256은 [manifest](../archive/source-manifest.json)에서 확인할 수 있습니다. 가중치는 `weights_only=True`로 읽으며, 모델 구조가 다른 파일은 호환되지 않습니다.

## 검증 범위와 한계

2026-10-11 로컬 CPU 검증에서 11개 단위 테스트를 통과했습니다. 합성 32×32 데이터로 데이터 검사, 1 epoch 학습, best/last 가중치 저장·로딩, 평가 JSON, 예측 PNG 생성을 확인했습니다. 별도 임시 환경에서 editable 설치도 검증했습니다. 원본 A4_2 가중치를 사용한 33×47 입력의 원본/정리본 logits는 동일했습니다(max absolute difference 0.0). 이는 구현과 가중치 호환성 검사입니다.

- README의 54.85%, 55.14%는 **수정본 발표 표에 남아 있는 validation mIoU**입니다. 이번 정리 과정에서 전체 학습·전체 검증을 다시 수행한 값이 아닙니다.
- 공개 예측 이미지는 보관된 A4_2 가중치로 정렬된 검증 샘플 중 고정 간격의 세 장을 다시 추론한 결과입니다. 선택 규칙과 환경은 [preview metadata](../results/preview-metadata.json)에 기록합니다. 일부 샘플의 정성 비교이며 전체 성능을 대신하지 않습니다.
- 원본에는 독립 test-set 결과, 여러 seed의 분산, 실차 영상 평가, 주행 안전성 또는 추론 속도 benchmark가 없습니다.
- 학습 데이터 규모가 작고 시뮬레이터 장면에 한정되어 있어 실제 도로 일반화는 추가 검증이 필요합니다.
- 단위 테스트와 호환성·추론 확인은 실행 결함을 확인하는 범위이며, 과거 성능 수치의 재현을 의미하지 않습니다.

## 다음 실험

1. 누락 라벨을 보완한 고정 split과 여러 seed로 기준 성능 재측정.
2. 클래스별 IoU를 기반으로 작은 객체와 희소 클래스의 오류 분석.
3. 증강, loss 구성, dropout을 한 번에 한 요소씩 바꾸는 통제 실험.
4. 별도 test split과 실제 도로 데이터에서 일반화 및 지연 시간 평가.
