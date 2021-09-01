---
layout: post
title: Optimal Control on EV EMS 프로젝트 소개
author: dongmin lee
categories: [optimal_control, reinforcement_learning]
image: assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_canvas.jpg
---

**[MakinaRocks](<http://www.makinarocks.ai>)의 HOS 팀**에서는 지난 2019년 9월부터 **전기 자동차(EV: Electric Vehicle)**에 들어가는 **에너지 관리 시스템(EMS: Energy Management System)**에 강화학습을 이용하여 최적 제어하는 프로젝트를 진행하고 있습니다.
HOS 팀은 글로벌 자동차 열 에너지 관리 솔루션 기업인 **[한온시스템(Hanon Systems)](<https://www.hanonsystems.com>)**과 함께 프로젝트를 진행 중이며, 연구 내용을 토대로 특허(마키나락스, 한온시스템. **특허출원 제10-2234270**, 2020) 및 [DEVIEW 발표](<https://tv.naver.com/v/16969158>) 등을 진행했습니다.

이번 글에서는 HOS 팀에서 풀고 있는 "**Optimal Control on EV EMS**" 프로젝트를 소개하고자 합니다!

## 풀고 있는 문제가 무엇인가?

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_canvas.jpg" alt="normal gradient" width="80%">
  <figcaption style="text-align: center;">[그림1] - 전기 자동차의 내부 모습</figcaption>
</p>
</figure>

전기 자동차는 석유 연료와 엔진을 사용하지 않고, 전기를 동력원으로 삼아 전기 배터리와 전기 모터를 사용하여 운행하는 자동차를 말합니다.
이러한 전기 자동차에는 에어컨, 배터리, 주행 등 다양한 곳에 전력 에너지가 쓰이는 것을 관리하는 에너지 관리 시스템이 존재합니다.
전기를 소모하는 에너지 관리 시스템은 전기 자동차의 주행 거리에 직접적인 영향을 미치게 되므로 시스템의 효율성이 매우 중요합니다.

<br>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_hvac.png" alt="normal gradient" width="80%">
  <figcaption style="text-align: center;">[그림2] - 공기 조화 시스템 [<a href="#ref-1">1</a>]</figcaption>
</p>
</figure>

에너지 관리 시스템 중에서도 에어컨과 관련된 냉방, 난방, 환기 등을 통합하여 자동차 환경의 안락을 위해 쓰이는 열 교환 시스템을 **공기 조화(HVAC: Heating, Ventilation, & Air Conditioning)** 시스템이라고 합니다.
공기 조화 시스템은 <span style="color:red">압축기, 팬, 밸브 등 시스템 안에 있는 다양한 부품들이 서로에게 영향을 주고 받는 시스템이며, 이러한 부품들을 이용하여 온도를 안전하고 효율적으로 조절하고 동시에 에너지 소모를 최소화하는 것을 목표</span>로 합니다. 

<br>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_pid_control.png" alt="normal gradient" width="60%">
  <figcaption style="text-align: center;">[그림3] - PID 제어 그래프</figcaption>
</p>
</figure>

공기 조화 시스템의 기존 제어 방법은 PID 제어를 사용합니다. PID 제어도 좋은 제어 방법이지만 아래와 같이 여러가지 단점들이 존재합니다. 
- 다변수 동시 제어의 어려움
- 선형 제어이기 때문에 비선형 시스템에서는 성능이 일정하지 않음
- 에너지의 효율 등의 다른 요소를 반영하기 어려움
- 일반적으로 단일 변수라 해도 최적 해가 아님
- 피드백 제어의 특성상 진동이 발생함

따라서 HOS 팀에서는 강화학습을 이용하여 기존 PID 제어보다 더 효과적이고 지능적인 제어 방법을 개발하여 공기 조화 시스템 문제를 해결하고자 합니다.
그렇다면 공기 조화 시스템 문제에 대한 강화학습 환경은 어떻게 정의하면 좋을까요?

## 강화학습 환경 정의

공기 조화 시스템은 차량의 필요에 따라 시스템(또는 circuit)을 다양한 모드로 변환하여 작동합니다.
HOS 팀에서는 다양한 모드 중 차량 cabin 내부의 **냉방 온도 제어**를 가정하고 강화학습 환경으로 정의하기 위해 환경에 어떠한 요소들이 포함되어 있는지 살펴보았습니다.
- **최종 목표**: 목표 온도까지 수렴하고 계속 유지하는 것 + 사용하는 에너지를 최소화하는 것
- **온도 수렴 task**: 초기 온도와 목표 온도가 주어졌을 때 사용하는 에너지를 최소화하여 초기 온도에서 목표 온도까지 빠르게 도달하는 것
- **온도 유지 task**: 목표 온도까지 도달했을 경우 사용하는 목표 온도에 계속 머무르면서 에너지를 최소화하는 것
- **다중 행동 요소**: 제어할 수 있는 여러 가지의 부품이 존재
- **비선형 시스템**: 공기 조화 시스템은 다양한 미분 방정식으로 이루어져 있으며, 비선형적인 계산으로 이루어져 있음

이러한 요소들을 고려하여 아래와 같이 상태, 행동, 보상을 정의하였습니다.

### 1. State representation

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_interaction_1.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림4] - 환경과 에이전트간의 상호작용</figcaption>
</p>
</figure>

강화학습에서의 중요한 특징 중 하나는 매 타임 스텝(time step)마다 에이전트와 환경간의 상호작용이 일어난다는 것입니다.
이 때 가장 먼저 생각해야할 문제는 환경에서 나온 다양한 상태 특징(feature) 중에서 에이전트한테 어떠한 상태 특징을 전달할 것인지에 대한 문제입니다.

HOS 팀에서는 특정 타임 스텝에서 받을 수 있는 정보와 이전 타임 스텝에서 얻을 수 있는 정보를 고려하여 아래와 같이 에이전트에게 전달할 기본적인 상태 특징들을 정의할 수 있었습니다.
- 이전 온도 값
- 현재 온도 값
- 현재 온도와 목표 온도의 차이
- 이전 행동 값
- 현재 온도가 목표 온도에 일정 값 이하로 도달 하였는지 유무

기본적인 상태 특징들 이외에도 다양한 상태 특징들이 존재하였기 때문에, 도메인 전문가분들의 의견과 팀 내부적으로 데이터 분석을 통해 사용하지 못하는 상태 특징들은 무엇인지, 시스템 delay를 고려하여 적절한 window size는 무엇인지 등을 알 수 있었습니다.
이렇게 알게된 상태 특징들을 추가적으로 상태 요소에 더하여 상태 표현을 완성하였습니다.

### 2. Action mapping

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_interaction_2.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림5] - 환경과 에이전트간의 상호작용</figcaption>
</p>
</figure>

강화학습에서 에이전트와 환경간의 상호작용을 할 때 두번째로 고려해야할 부분은 에이전트가 현재 상태를 보고 어떠한 행동을 하였을 때 환경에서 이 행동을 잘 받아들일 수 있도록 만들어주어야 합니다.
쉽게 말해, 일반적으로 에이전트의 행동을 담당하는 policy network는 [-1, 1]의 값으로 출력하기 때문에 출력된 값을 환경으로 넣어줄 때는 환경에서 원하는 범위로 mapping을 해주어야 원활한 상호작용이 이루어집니다.

본 프로젝트 문제는 환경에서 원하는 범위로 mapping할 때 **slew rate**이라는 행동에 대한 하드웨어의 제약이 존재합니다.
slew rate이란 아래의 그림처럼 실제 차량의 장비 구동이 일정 이상 증감이 불가한 물리적인 제약을 의미하며, 이전 행동에 따라 다음에 실행할 수 있는 행동 변화의 한계치를 말합니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_slew_rate.png" alt="normal gradient" width="50%">
  <figcaption style="text-align: center;">[그림6] - 시간 대비 slew rate를 이용한 행동 그래프</figcaption>
</p>
</figure>

HOS 팀에서는 slew rate를 반영하여 행동을 mapping하는 방법으로 범위 외 값을 제거하는 clipping 방법과 범위를 재조정하는 rescaling 방법을 사용하였습니다.
Clipping 방법은 policy network에서 나온 [-1, 1]값을 모든 행동 범위로 mapping한 후 가능한 범위로 자르는 방법이고, rescaling 방법은 이전 행동을 기준으로 가능한 행동의 범위 안에서 값을 내는 방법입니다.
학습과정에서는 큰 차이가 없었지만, clipping 방법과는 달리, rescaling 방법은 진동하는 경향이 없었기 때문에 더 안전하고 효율적인 제어가 가능했습니다.

### 3. Reward function

머신러닝에서 다른 학습 방법과는 달리 강화학습만의 중요한 특징은 바로 보상 함수를 통해 에이전트가 학습한다는 것입니다.
에이전트가 학습할 수 있는 유일한 요소이기 때문에 보상 함수를 잘 구성해야합니다.
또한 기존 PID 제어는 다양한 요구 조건을 유연하게 반영할 수 없는 반면, 강화학습은 온도, 에너지 효율, 수렴 속도 등 여러 요소를 아래와 같이 보상 함수를 만들어 이를 에이전트가 학습할 수 있다는 장점이 있습니다.

$$
\eqalign{
r(s,a) = \sum r_k = r_{\text{temperature}} + r_{\text{efficiency}} + r_{\text{time}} + \cdots
}
$$

<br>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_temperature.png" alt="normal gradient" width="60%">
  <figcaption style="text-align: center;">[그림7] - 시간 대비 온도 그래프</figcaption>
</p>
</figure>

본 프로젝트 문제에서는 위의 그림처럼 목표 온도까지 빠르게 도달해야하는 온도 가변 구간과 도달한 목표 온도를 계속 유지해야하는 온도 유지 구간으로 나누어 볼 수 있습니다.
두 구간의 요구 조건이 다르기 때문에 요구 조건에 따른 보상 함수를 각각 다르게 주어야 했습니다.

<br>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_temperature_variable_interval.png" alt="normal gradient" width="100%">
  <figcaption style="text-align: center;">[그림8] - 온도 가변 구간 그래프</figcaption>
</p>
</figure>

먼저, 온도 가변 구간의 목표는 목표 온도에 빠르게 수렴하고, 에너지의 양을 최소화하는 것입니다. 이에 따라 음수 보상과 양수 보상을 반영하였을 때 위의 그림과 같은 결과가 나왔습니다.
양수 보상이 목표 온도에 더 빠르게 수렴하는 양상을 보였지만, 에너지의 소모가 매우 크기 때문에 양수 보상보다 음수 보상을 선택하였습니다.

<br>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_temperature_convergence_interval.png" alt="normal gradient" width="100%">
  <figcaption style="text-align: center;">[그림9] - 온도 유지 구간 그래프</figcaption>
</p>
</figure>

다음으로, 온도 유지 구간의 목표는 도달한 목표 온도를 유지하고, 에너지의 양을 최소화하는 것입니다.
위의 그림과 같이 위와 반대로 음수 보상이 오히려 진동하는 경향을 보였습니다.
따라서 온도 유지 구간에서는 양수 보상을 선택하였습니다.

지금까지 공기 조화 시스템 문제에 대해 HOS 팀에서 정의한 강화학습 환경 요소인 상태, 행동, 보상에 대해 살펴봤습니다.
강화학습 환경 정의에 대한 더 자세한 내용은 [DEVIEW 2020](<https://tv.naver.com/v/16969158>) 영상을 참고해주시기 바랍니다.

## 강화학습 에이전트 학습시키기

이전 내용에서는 공기 조화 시스템에 대한 환경을 정의하는 것에 대해 알아봤습니다.
환경에 대한 정의는 잘 마쳤지만, 정의된 상태, 행동, 보상을 바탕으로 강화학습 에이전트를 학습시키기 위해서는 다음과 같은 문제들을 해결해야 했습니다.
- 공기 조화 시스템의 시뮬레이터를 이용하여 강화학습 환경을 구성하는 문제
- 강화학습 알고리즘을 구현하여 사용할 때의 알고리즘 성능의 신뢰도 문제
- Windows OS에 대한 시뮬레이터의 라이센스와 강화학습 알고리즘으로 사용하는 라이브러리의 인스턴스 문제

각각의 문제에 대해 HOS 팀에서 해결했던 과정들을 소개해보려 합니다.

### 1. FMU

강화학습 환경을 구성하는 방법에는 어떤 방법들이 있을까요?
일반적으로, 시뮬레이터를 만들 수 있는 물리 엔진(e.g., Unity3D)을 이용하여 직접 개발한 시뮬레이터에서 강화학습 환경을 구성하는 방법과 기존에 구현된 시뮬레이터를 활용하여 강화학습 환경을 구성하는 방법이 있습니다.
전자의 경우, 공기 조화 시스템의 시뮬레이터를 직접 만들기에는 도메인 지식을 이용하여 시스템에 담긴 다양한 비선형적 계산들을 구현해야하기 때문에 실제로 만드는 것이 굉장히 어렵습니다.
따라서 HOS 팀에서는 클라이언트 측에 있는 산업용 시뮬레이터를 이용하여 강화학습 환경을 구성하기로 했습니다.

하지만 환경을 구성하기 위해서는 시뮬레이터가 클라이언트 측에 있는 windows OS에 설치되어 있었기 때문에 보안 문제로 인해 블랙박스 형태인 dynamic system 모델로 이용해야 했습니다. 또한 강화학습 에이전트를 학습시키기 위해서는 파이썬 연동을 해야했습니다.
HOS 팀에서는 이러한 문제들을 FMI format으로 추출된 시뮬레이터 파일과 해당 format에 사용할 수 있는 파이썬 라이브러리인 FMPy를 사용하여 해결할 수 있었습니다.

<br>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_fmi.png" alt="normal gradient" width="80%">
  <figcaption style="text-align: center;">[그림10] - FMI [<a href="#ref-2">2</a>]</figcaption>
</p>
</figure>

먼저, **FMI (Functional Mock-up Interface)**[[2](#ref-2)]란, XML 파일과 C언어 코드의 조합을 통해 하나의 파일에 압축되어 dynamic system 모델을 연동하기 위해 블랙박스 형태의 컨테이너와 인터페이스를 제공하는 자유 표준을 말합니다.
FMI는 FMI의 호환 모델을 포함하고 있는 하나의 인스턴스인 FMU (Function Mock-up Unit)로 시뮬레이터의 모델을 표준화하고 이를 접근하기 위한 인터페이스를 정의합니다.
이로 인해 시뮬레이션에 사용되는 모든 구성 요소의 도메인 지식을 모르더라도 FMU를 통해 시뮬레이션을 디자인하는 것이 가능하게 됩니다.

다음으로, FMU를 파이썬 연동 및 시뮬레이션하기 위해 파이썬 라이브러리인 **FMPy**를 이용하였습니다.
FMPy[[3](#ref-3)]에 있는 함수들을 통해 만들어진 FMU를 학습이 시작될 때 불러오고, 에피소드마다 초기화하고, 학습이 끝났을 때 종료할 수 있었습니다.
또한 FMU에 에이전트의 행동을 넣어주고, 한 타임 스텝동안 시뮬레이션을 진행하고, 다음 상태를 뽑아주는 에이전트와 환경간의 상호작용을 원활하게 나타낼 수 있었습니다. 

### 2. Ray RLlib

HOS 팀에서는 기존에 자체 제작된 MakinaRocks의 파이썬 강화학습 프레임워크인 RLocks를 사용하였습니다.
하지만 강화학습 알고리즘을 구현한 코드에 대해 계속적으로 검증해야하기 때문에 많은 시간을 소모해야하고, 무엇보다도 알고리즘에 대한 성능의 신뢰도 문제가 생기게 됩니다. 

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_ray_rllib_stack.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림11] - Ray RLlib stack [<a href="#ref-4">4</a>]</figcaption>
</p>
</figure>

이에 HOS 팀에서는 강화학습의 멀티 에이전트 학습과 분산 학습이 가능하도록 높은 scalability와 통합된 API를 제공하는 오픈소스 라이브러리인 **Ray RLlib**[[4](#ref-4)]을 사용하였습니다.
Ray RLlib을 통해 알고리즘에 대한 검증과 신뢰도 문제를 해결할 수 있었으며, 다양한 알고리즘 중 멀티 에이전트 Soft Actor-Critic (SAC) 알고리즘을 이용하여 FMU 환경에서 학습을 진행하였습니다.

### 3. Docker를 이용한 TCP 통신

FMU를 통해서 강화학습 환경을 구성하고, Ray RLlib을 통해서 강화학습 알고리즘을 돌렸지만, 한가지 더 문제가 있었습니다.
클라이언트 측에 있는 windows OS에서 Ray RLlib을 사용할 때 ray 인스턴스를 2개 이상 띄울 수 없는 문제가 생겼습니다.
클라이언트 측에서 전달받은 산업용 시뮬레이터 라이센스는 2개지만, 띄울 수 있는 ray 인스턴스는 1개만 가능했기 때문에 독립적인 학습이 불가능했습니다.
Window OS의 ray 인스턴스 문제에 대한 자세한 내용은 [링크](<https://github.com/ray-project/ray/issues/9265>)를 참고하시기 바랍니다.

이를 해결하기위해 HOS 팀에서는 **Docker**[[5](#ref-5)]의 linux contrainer와 FMU가 있는 windows host간의 **TCP 통신**을 통해 환경과 에이전트가 간접적으로 상호작용하는 방식을 개발하였습니다.
여기서 docker란 어플리케이션을 신속하게 구축, 테스트 및 배포할 수 있는 소프트웨어 플랫폼을 말합니다.
Docker는 소프트웨어를 컨테이너라는 표준화된 유닛으로 패키징하며, docker를 사용하면 개발 환경에 구애받지 않고 어플리케이션을 신속하게 배포 및 확장할 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_interaction_using_docker_and_tcp.png" alt="normal gradient" width="100%">
  <figcaption style="text-align: center;">[그림12] - Docker와 TCP 통신을 이용한 에이전트와 환경간의 상호작용</figcaption>
</p>
</figure>

TCP 통신을 이용하여 학습하는 방식의 중요 요소로는 크게 Ray RLlib, Pseudo env, Env로 나눌 수 있습니다.
위의 그림에서 먼저 linux container에서 에이전트 역할을 하는 **Ray RLlib**은 시작할 때나 에피소드가 종료되었을 때 환경에 reset을 요청하고 결과값인 상태를 받습니다.
또한 에피소드 중간에 행동을 전달하고 다음 상태, 보상, 에피소드 종료의 유무, 환경에 대한 정보를 받습니다.

다음으로, linux container에서 환경 역할을 하는 **Pseudo env**는 windows host의 Env와 TCP 통신을 하며, 에이전트가 step, reset method를 사용하면 windows host의 env에 요청을 보냅니다.
Pseudo env는 기존 학습코드에서 환경만 대체하게 되므로 기존 코드를 변경하지 않고도 호환이 가능합니다.

마지막으로, windows host의 **Env**는 container에서 들어오는 요청에 반응하도록 TCP wrapper를 만들어 줍니다.
TCP wrapper가 된 env는 요청에 따라 FMU 환경에 있는 step, reset method를 진행하고 이에 따른 정보를 container로 전달합니다.

HOS 팀에서는 이러한 방법을 통해 windows OS에서 Ray RLlib을 사용할 때 ray 인스턴스를 2개 이상 띄울 수 없는 문제를 해결할 수 있었습니다.

## 성능과 한계점, 그리고 이를 해결하기 위한 방법론

HOS 팀에서는 이전 내용과 같이 다양한 해결방법을 통해 공기 조화 시스템에서 강화학습 알고리즘을 이용한 제어가 기존 PID 제어보다 에너지 효율성을 최대 20%까지 향상시키는 성과를 확인하였습니다.
또한 기존 제어 방법에 필요한 대체 목표값을 강화학습을 이용하여 찾을 수 있었으며, 강화학습을 이용한 제어 방법과 기존 제어 방법간의 융합을 통한 문제 해결의 가능성을 확인할 수 있었습니다.

HOS 프로젝트의 최종 목표는 실제 전기 자동차의 공기 조화 시스템을 제어하는 것입니다.
하지만 시뮬레이터에서 학습된 제어 시스템을 실제 전기 자동차에 탑재할 경우 매우 치명적인 문제가 생깁니다.
바로 공기 조화 시스템에 대한 시뮬레이터와 실제 자동차간의 간극이 존재하는 문제입니다.
따라서 새롭게 문제는 **"시뮬레이터가 아닌 실제 시스템에서 어떻게 기존 제어 방법보다 더 효율적인 제어 방법을 만들 수 있을까?"**로 넘어오게 됩니다. 

<br>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_sim2real.png" alt="normal gradient" width="49%"> <img src="/assets/images/2021-06-02-optimal_control_on_ev_ems_project/optimal_control_on_ev_ems_offline_rl.gif" alt="normal gradient" width="49%">
  <figcaption style="text-align: center;">[그림13] - Sim-to-Real & Offline Reinforcement Learning [<a href="#ref-6">6</a>, <a href="#ref-7">7</a>]</figcaption>
</p>
</figure>

한가지 다행인 것은 실제 전기 자동차의 공기 조화 시스템을 이용한 실제 데이터를 얻을 수 있었습니다.
실제 데이터를 이용한 방법들을 찾아봤을 때, 시뮬레이터에서 학습한 모델(e.g., policy network, dynamics model)을 어떻게 실제 시스템으로 옮길 수 있는지에 대한 **Sim-to-Real (Simulation to Real world)** 방법과 실제 데이터를 가지고 offline learning 방법을 이용하는 **Offline Reinforcement Learning** 방법을 생각해볼 수 있었습니다.
Sim-to-Real과 Offline RL에 대한 자세한 내용은 아래의 related posts를 참고해주시기 바랍니다.

## 마치며

이번 포스팅에서는 MakinaRocks의 HOS 팀에서 공기 조화 시스템이라는 실제 문제에 강화학습을 이용하여 최적 제어를 할 때 접했던 다양한 문제들과 이에 대해 해결했던 과정들을 소개하였습니다.
현재 HOS 팀은 시뮬레이터가 아닌 실제 전기 자동차의 공기 조화 시스템에서 강화학습을 이용하여 기존 제어 방법보다 더 효율적인 제어 방법을 만들기 위해 계속해서 연구 개발을 진행하고 있습니다.

이번 포스팅이 강화학습을 이용하여 실제 문제에 적용하시는 분들께 많은 도움이 되었으면 좋겠습니다. 감사합니다!

## Related Posts

**Sim-to-Real**

본 프로젝트에서 시뮬레이터와 실체 자동차간의 간극을 줄이기 위한 방법인 sim-to-real에 대해 대표적인 논문들을 선정하여 회사 내에서 발표를 진행했습니다. 자료는 아래의 링크를 통해 확인하실 수 있습니다. 
- [Sim-to-Real (presented by Dongmin Lee)](<https://www.slideshare.net/DongMinLee32/simtoreal>)

**Offline Reinforcement Learning**

HOS 팀은 실제 차량의 데이터를 가지고 강화학습을 하기 위한 여러가지 방법 중 하나인 offline RL을 적용하고 있습니다.
Offline RL 논문 중 "Conservative Q-Learning (CQL)"이라는 논문을 아래의 포스팅으로 정리해봤습니다.
- TBU
<!-- [Conservative Q-Learning for Offline Reinforcement Learning (written by Dongmin Lee)]() -->

## References

<a name="ref-1">[1]</a>  [François-Xavier Keller, "Virtual Thermal Comfort Engineering", Technical Report, 2003](https://www.researchgate.net/publication/307173941_Virtual_Thermal_Comfort_Engineering)

<a name="ref-2">[2]</a>  ["Functional Mock-up Interface (FMI)" website](https://fmi-standard.org/)

<a name="ref-2">[3]</a>  ["FMPy" documentation](https://fmpy.readthedocs.io/en/latest/)

<a name="ref-3">[4]</a>  ["Ray RLlib" documentation](https://docs.ray.io/en/master/rllib.html)

<a name="ref-3">[5]</a>  ["Docker" documentation](https://docs.docker.com/)

<a name="ref-4">[6]</a>  [Yevgen Chebotar, Ankur Handa, Viktor Makoviychuk, Miles Macklin, Jan Issac, Nathan Ratliff, Dieter Fox, "Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience", preprint, 2018](https://arxiv.org/pdf/1810.05687.pdf)

<a name="ref-5">[7]</a>  [Google Research, "An Optimistic Perspective on Offline Reinforcement Learning", Google AI Blog, 2020](https://ai.googleblog.com/2020/04/an-optimistic-perspective-on-offline.html)
