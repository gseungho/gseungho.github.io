---
title: "졸음 운전 감지 시스템을 직접 만들어보며 배운 것들"
date: "2026-05-20"
category: "프로젝트"
tags: ["딥러닝", "LSTM", "MediaPipe", "컴퓨터비전"]
description: "MediaPipe + LSTM + Attention으로 실시간 졸음 운전 감지 시스템을 구현한 딥러닝 수업 팀 프로젝트 회고"
---

졸음 운전은 반응 속도 저하와 집중력 감소를 동반해 심각한 사고로 이어진다. 국내 경찰청 통계에 따르면 졸음 운전 치사율은 음주 운전 치사율을 상회한다. 그런데 시중의 졸음 감지 솔루션은 대부분 고가의 적외선 센서나 웨어러블 기기를 요구한다.

"웹캠 하나로 되지 않을까?" 라는 질문에서 이 프로젝트가 시작됐다.

딥러닝 수업 팀 프로젝트로, 5명이 약 한 달 반 동안 MediaPipe와 LSTM 기반의 실시간 졸음 감지 시스템을 만들었다. 결과적으로 테스트셋 기준 **Recall 91.53%, Accuracy 87.83%** 를 달성했다. 이 글은 그 과정에서 시도한 것들, 실패한 것들, 그리고 실제로 효과가 있었던 것들에 대한 기록이다.

---

## 데이터를 직접 만들었다

데이터셋은 두 가지를 합쳤다.

**Kaggle NTHU-DDD**: 졸음 운전 감지 연구에서 많이 쓰이는 공개 데이터셋이다. 다양한 피험자의 졸음/정상 영상이 포함돼 있다. 30fps 영상을 10fps로 다운샘플링해 프레임 이미지로 추출했다.

**자체 제작 데이터**: Kaggle 데이터만으로는 촬영 각도와 환경이 제한적이라고 판단했다. 팀원들이 직접 졸음 상태와 정상 상태를 연출해 영상을 촬영하고, 마찬가지로 10fps로 추출해 레이블링했다.

아래 두 이미지가 자체 제작 데이터의 실제 샘플이다. 왼쪽이 졸음(Drowsy), 오른쪽이 정상(Not Drowsy) 상태다.

<div style="display: flex; gap: 16px; justify-content: center; margin: 24px 0;">
  <figure style="text-align: center; margin: 0;">
    <img src="/images/posts/drowsy-detection/sample-drowsy.jpg" alt="졸음 상태 샘플" style="width: 200px; border-radius: 8px;" />
    <figcaption style="margin-top: 8px; color: #888; font-size: 0.9em;">Drowsy — 눈이 감기고 고개가 숙여진 상태</figcaption>
  </figure>
  <figure style="text-align: center; margin: 0;">
    <img src="/images/posts/drowsy-detection/sample-notdrowsy.jpg" alt="정상 상태 샘플" style="width: 200px; border-radius: 8px;" />
    <figcaption style="margin-top: 8px; color: #888; font-size: 0.9em;">Not Drowsy — 눈을 뜨고 정면을 바라보는 상태</figcaption>
  </figure>
</div>

파일명 규칙은 `{인물ID}_{라벨}({프레임번호}).jpg` 형태다. 인물별로 묶어서 처리해야 Nod Calibration 같은 개인별 보정이 가능하다.

전체 학습 데이터는 28,245 프레임이며, 이를 Sliding Window(길이 40, stride 5)로 자르면 수천 개의 시퀀스가 만들어진다.

---

## 이미지를 픽셀로 넣지 않은 이유

가장 먼저 맞닥뜨린 질문은 **"어떤 입력을 모델에 넣을 것인가"** 였다.

영상에서 졸음을 분류하는 방법은 크게 두 가지다.

1. 원본 프레임을 CNN에 통과시켜 특징을 뽑는 방법 (2D CNN, 3D CNN)
2. 얼굴 랜드마크를 먼저 추출해 수치 특징으로 변환한 뒤 시계열 모델에 넣는 방법

우리는 후자를 선택했다. 이유는 간단했다. CNN 기반 접근은 조명, 카메라 해상도, 배경 등 이미지 레벨 노이즈에 취약하고, 학습에 필요한 데이터 양도 훨씬 많다. 반면 랜드마크 기반 특징은 해상도와 무관하게 일관된 수치를 뽑을 수 있고, 무엇보다 "왜 졸음으로 판단했는가"를 해석하기 쉽다.

[MediaPipe FaceMesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)는 얼굴의 468개 랜드마크를 실시간으로 추출해준다. 여기서 7개의 특징을 계산했다.

| 특징 | 설명 |
|------|------|
| EAR | Eye Aspect Ratio. 눈의 세로/가로 비율. 눈이 감길수록 0에 가까워진다 |
| MAR | Mouth Aspect Ratio. 입 벌림 정도. 하품 감지에 활용 |
| Nod | 고개 끄덕임 정도 (코끝과 턱의 y축 거리 비율) |
| Tilt | 머리 기울기 (좌우 귀 연결선의 각도) |
| d_EAR | EAR의 프레임 간 변화량 (눈 깜빡임 속도) |
| PERCLOS | 최근 10프레임 내 눈 감김 비율 |
| d_MAR | MAR의 프레임 간 변화량 |

초기엔 EAR, MAR, Nod 3개만 사용했다. 나중에 4개를 추가했는데, 그 이야기는 뒤에서 다룬다.

---

## 모델 구조: LSTM + Self-Attention

입력은 **40프레임 × 7특징** 시퀀스다. 10 FPS 기준으로 4초 분량의 얼굴 움직임 패턴이다. 이걸 Sliding Window 방식(stride=5)으로 잘라서 학습 데이터를 만들었다.

모델 구조는 다음과 같다.

```python
inputs = Input(shape=(40, 7))

x = LSTM(64, return_sequences=True)(inputs)
x = LayerNormalization()(x)
x = Dropout(0.5)(x)

x = LSTM(32, return_sequences=True)(x)
x = LayerNormalization()(x)
x = Dropout(0.5)(x)

att = Attention()([x, x])
merged = Concatenate()([x, att])
x = GlobalAveragePooling1D()(merged)

x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
```

LSTM이 시계열 패턴을 잡고, Self-Attention이 40프레임 중 어떤 시점의 특징이 중요한지에 가중치를 준다. 졸음은 순간적인 현상이 아니라 누적되는 패턴이기 때문에 이 구조가 적합하다고 판단했다.

---

## 삽질의 기록: 처음엔 잘 안 됐다

### Dropout 0.3이 문제였나?

초기 모델은 Dropout 0.3으로 시작했다. 검증셋 성능이 불안정했다. 0.3, 0.5, 0.7을 비교 실험한 결과 **0.5가 가장 안정적**이었다. 0.7은 과소적합 경향이 나타났다.

### 특징 3개로는 부족했다

EAR, MAR, Nod만으로는 졸음의 복합 패턴을 포착하기 어려웠다. 특히 "순간적으로 눈을 감은 것"과 "지속적으로 눈이 감겨 있는 것"을 구분하지 못했다.

Tilt(머리 기울기), d_EAR(EAR 변화 속도), PERCLOS(지속 눈 감김 비율), d_MAR(MAR 변화 속도)를 추가해 7개로 늘렸다. 특히 PERCLOS는 졸음 연구에서 오래전부터 사용되던 지표로, 단순 순간 눈 감김이 아니라 지속적인 눈 감김 비율을 측정한다.

### Nod 값이 예상치 못하게 튀었다

가장 골치 아팠던 문제였다. 실패한 예측 영상들의 특징값을 시각화하다가 발견했는데, Nod 값이 특정 프레임에서 비정상적으로 크게 튀는 현상(spike)이 있었다.

원인은 두 가지였다. 랜드마크 추출 실패, 그리고 카메라 앵글에 따른 왜곡. 모델은 이 spike를 "강한 졸음 신호"로 오인했고, 멀쩡히 운전하는 영상을 졸음으로 분류했다.

두 가지로 해결했다.

**Safety Clipping**: Nod, Tilt 값을 $[-3.0, 3.0]$ 범위로 강제 클리핑해 노이즈를 제거했다.

**Nod Calibration**: 영상 초반 3초의 평균 Nod 값을 기준점(0점)으로 설정했다. 사람마다 카메라와의 거리나 앉는 자세가 다르기 때문에, 절대값이 아니라 개인의 평상시 자세 대비 변화량으로 측정하도록 바꿨다.

```python
# 초반 30프레임(3초)을 기준으로 보정
if len(features_arr) > 10:
    nod_vals = features_arr[:, 2]
    valid_mask = nod_vals[:30] != 0
    if np.sum(valid_mask) > 5:
        baseline_nod = np.mean(nod_vals[:30][valid_mask])
        non_zero_mask = nod_vals != 0
        features_arr[non_zero_mask, 2] -= baseline_nod
```

이 처리 이후 오탐지율이 눈에 띄게 줄었다.

### Class Weight가 양날의 검이었다

Drowsy 클래스에 1:1.2 가중치를 주자 Recall이 크게 올랐지만, 동시에 정상 영상을 졸음으로 오분류하는 False Positive도 함께 증가했다. 가중치를 1.4, 1.6으로 올릴수록 이 현상은 심해졌다.

결국 Class Weight를 걷어내고 대신 **Focal Loss**를 도입했다.

### 손실 함수를 BCE에서 Focal Loss로

Focal Loss의 핵심은 동적 가중치다.

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

모델이 이미 잘 맞추는 샘플은 $(1 - p_t)^\gamma$ 항이 0에 수렴해 손실 기여가 거의 없어진다. 반대로 틀리기 쉬운 샘플에는 상대적으로 높은 손실 가중치가 유지된다. Class Weight처럼 특정 클래스를 억지로 강조하는 게 아니라, 난이도 기반으로 자동 조절된다.

$\gamma$ 값별 실험 결과:

| Gamma | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| 2.0 | 0.6724 | 0.6375 | 0.8500 | 0.7286 |
| 3.0 | 0.8448 | 0.8750 | 0.8167 | 0.8448 |
| 4.0 | 0.8103 | 0.7794 | 0.8833 | 0.8281 |
| 5.0 | 0.8448 | 0.8500 | 0.8500 | 0.8500 |

$\gamma = 3.0 \sim 5.0$ 구간에서 성능이 안정적이었다. 최종 모델에는 $\gamma = 2.0$에 **Label Smoothing 0.1**을 추가했다. Label Smoothing은 정답을 0/1이 아닌 0.1/0.9 수준으로 부드럽게 설정해 모델이 지나치게 확신하는 것을 방지한다. 높은 확률로 틀리는 오탐지를 줄이는 데 효과적이었다.

### LSTM 레이어 크기 비교

| 레이어 크기 | Dropout | Accuracy | Recall | F1 | AUC |
|------------|---------|----------|--------|----|-----|
| (128, 64) | 0.5 | 0.851 | 0.923 | 0.867 | 0.887 |
| **(64, 32)** | **0.7** | **0.869** | **0.938** | **0.891** | **0.917** |
| (32, 16) | 0.7 | 0.847 | 0.939 | 0.876 | 0.901 |

**(64, 32)** 구성이 가장 균형 잡힌 성능을 보였다. 레이어가 클수록 무조건 좋지 않았다. 과적합 방지와 일반화 사이의 균형이 중요했다.

---

## 실시간 감지: 웹캠에 모델을 붙이다

학습이 끝난 뒤, 저장된 `.h5` 모델을 불러와 웹캠 영상에 실시간으로 연결했다. 구조는 단순하다.

1. 웹캠 프레임을 MediaPipe로 처리해 7개 특징값 추출
2. 길이 40의 슬라이딩 버퍼에 프레임별 특징을 쌓는다
3. 버퍼가 꽉 차면 LSTM 모델에 넣어 졸음 확률 계산
4. 결과를 화면에 오버레이

```python
feature_buffer = deque(maxlen=SEQUENCE_LENGTH)  # 40프레임 버퍼

while cap.isOpened():
    ret, frame = cap.read()
    feats, prev_ear, prev_mar, history_ear = _extract_single_frame_features(
        frame, prev_ear, prev_mar, history_ear
    )
    if feats is not None:
        feature_buffer.append(feats)

    if len(feature_buffer) == SEQUENCE_LENGTH:
        X = normalize_features(np.array(feature_buffer))
        prob = model.predict(X[np.newaxis, ...])[0][0]  # 졸음 확률
        label = "DROWSY" if prob > 0.6 else "Normal"
```

실시간으로 동작하는 모습은 아래 영상에서 확인할 수 있다.

<video src="/images/posts/drowsy-detection/demo.mp4" controls width="100%" />

영상에서 눈에 띄는 특징이 하나 있다. 하품을 하거나 눈을 감아도 졸음 퍼센트가 즉시 올라가지 않고 **약 2초 뒤에 반응**한다. 의도한 동작이다. 모델이 40프레임(4초)의 누적 패턴을 보고 판단하기 때문에, 입력이 변화하더라도 버퍼에 졸음 신호가 충분히 쌓인 뒤에야 확률이 올라간다. 실제로 이 지연 덕분에 순간적인 눈 깜빡임이나 잠깐의 고개 숙임을 졸음으로 오판하는 경우가 줄었다.

반대로 보면 반응 속도가 느린 것이기도 하다. 진짜 졸음 상황에서 2~4초 뒤에 경고가 울린다면 의미가 있는가에 대한 질문은 여전히 유효하다.

또 다른 문제는 **프레임 끊김**이다. 영상에서 보이듯 실시간 처리 중 화면이 미세하게 버벅인다. MediaPipe와 모델 추론이 메인 스레드에서 돌고 있어서 생기는 문제다. 실제 차량 탑재를 목표로 한다면 두 가지가 필요하다.

- **경량화**: 양자화(Quantization) 또는 지식 증류(Knowledge Distillation)로 모델 크기를 줄여 추론 속도를 높인다.
- **멀티스레딩**: 영상 캡처, 특징 추출, 모델 추론을 별도 스레드로 분리해 프레임 드롭을 방지한다.

---

## 최종 결과

테스트셋(119 clips, 학습에 전혀 사용하지 않은 영상들)에서의 성능:

| 지표 | 값 |
|------|----|
| Accuracy | **87.83%** |
| Recall | **91.53%** |
| Precision | **85.71%** |
| F1-Score | **0.88** |

졸음 감지 시스템에서 Recall을 최우선으로 설정한 이유는 명확하다. 졸지 않는 사람에게 경고를 보내는 것(낮은 Precision)은 불편함이지만, 졸고 있는 사람을 놓치는 것(낮은 Recall)은 사고다. 91.53%의 Recall은 실제 졸음 상태의 약 11명 중 10명을 감지한다는 의미다.

---

## 여전히 남아있는 문제들

**데이터 다양성 부족**: 학습 데이터 대부분이 잘 통제된 실내 환경에서 촬영됐다. 선글라스 착용, 야간 촬영, 여러 인종과 얼굴형에 대한 검증이 부족하다.

**실험 재현성**: 같은 설정으로 여러 번 학습해도 지표가 편차를 보였다. 테스트셋 크기(119 clips)가 작아서, 0.4~0.6 확률로 예측하는 경계선상의 샘플 몇 개가 전체 지표를 크게 흔든다.

**실시간 성능**: PC 환경에서도 프레임이 미세하게 끊겼다. 차량 내 임베디드 시스템에 탑재하려면 양자화(Quantization) 같은 경량화 작업이 필요하다.

**다중 탑승자 처리**: 카메라에 두 명이 잡히는 경우, 아무도 없는 경우에 대한 처리 로직이 없다.

---

## 돌아보며

이 프로젝트에서 가장 많이 배운 건 모델 아키텍처보다 **데이터를 직접 들여다보는 습관**이었다.

Nod spike 문제는 손실 함수나 레이어 크기를 아무리 바꿔도 해결이 안 됐다. 실제로 틀린 예측의 특징값을 시각화하고 나서야 원인을 찾았다. 모델이 왜 틀렸는지 모르면 운에 의존해서 파라미터를 바꾸는 것과 다르지 않다.

Safety Clipping과 Nod Calibration이라는 단순한 전처리 두 줄이 Focal Loss 튜닝보다 훨씬 큰 성능 향상을 가져왔다.

프로젝트 코드는 [GitHub](https://github.com/gseungho)에서 확인할 수 있다.
