---
layout: post
title: 오픈소스(RLlib) 문제 발견부터 컨트리뷰션 까지
author: yuri rocha
categories: [open_source, reinforcement_learning]
image: assets/images/2021-08-20-open-source-contributions-rllib/OLP_example_image.jpg
---

[마키나락스](http://www.makinarocks.ai/)의 OLP(Off-line Programming) 팀에서는 제조 공장에서 사용되는 Multi-Robot Arm의 경로계획(Path Planning) 문제를 강화학습을 이용하여 풀고 있습니다.
(*경로계획이란 다수의 로봇팔들이 효과적으로 동작할 수 있는 경로를 생산하는 문제입니다.*)
여러 시행착오를 거치며 학습하는 강화학습 모델의 특성 때문에, 시간, 안전, 비용의 문제가 발생할 수 있는 실제 로봇을 사용하는 대신 시뮬레이터를 이용하여 모델을 학습시키고 있습니다.
이때 강화학습에 적용할 수 있는 시뮬레이터를 만드는 과정에서 몇 가지 오픈소스 소프트웨어(오픈소스)를 사용하게 되었습니다.

본 포스팅에서는 오픈소스([RLlib](https://docs.ray.io/en/latest/rllib.html))를 사용하며 발견한 문제의 원인 분석부터 컨트리뷰션을 통한 문제해결까지의 과정을 공유드리려고 합니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-20-open-source-contributions-rllib/OLP_example_image.jpg" alt="OLP" width="80%">
  <figcaption style="text-align: center;">[그림1] Multi-Robot Arm 실제 환경</figcaption>
</p>
</figure>

# 오픈소스 컨트리뷰션

오픈소스는 공개적으로 액세스할 수 있게 설계되어 누구나 자유롭게 확인, 수정, 배포할 수 있는 코드입니다.
오픈소스의 핵심은 접근성이 뛰어난 가치 있는 오픈소스를 만드는 것이지만, 이러한 프로젝트에 기여하는 것은 컨트리뷰터에게 또한 도움이 될 수 있습니다.
오픈소스를 유지, 관리하며 기술적인 능력을 연마할 수 있을 뿐 아니라, 오픈소스 커뮤니티의 발전에도 기여할 수 있기 때문입니다.
더구나, 회사 관점에서는 회사에 적용하는 오픈소스를 개선, 인원의 기술력 성장, 대외적 기술 인지도 향상 등 다양한 장점도 있습니다.

# 컨트리뷰션 배경

## ML-Agents


[ML-Agents](https://github.com/Unity-Technologies/ml-agents)는 Unity3D 게임 엔진을 기반으로 작성된 시뮬레이터로서, 지능형 에이전트의 학습 환경으로 사용되는 오픈소스입니다.
([지난 블로그 포스팅](https://makinarocks.github.io/Building-a-Reinforcement-Learning-Environment/)에서는 OLP 프로젝트를 위해 정의한 Unity3D 환경에 대해 소개해 드린 적이 있습니다.)

## Ray RLlib

[Ray](https://github.com/ray-project/ray)는 파이썬 기반의 분산처리 프레임워크로, 다양한 기계학습 라이브러리를 포함하고 있습니다. 그 중 RLlib은 확장 가능한 (Scalable) 분산 강화학습 환경을 간편히 구축할 수 있도록 도와주는 라이브러리입니다.
OLP팀에서는 RLlib을 사용하여 100 개 이상의 worker(독립적인 환경)를 사용하여 강화학습을 수행하고 있습니다.
RLlib은 기본적으로 Unity3D의 ML-Agents 환경에 연결할 수 있는 [Wrapper](https://medium.com/distributed-computing-with-ray/reinforcement-learning-with-rllib-in-the-unity-game-engine-1a98080a7c0d)을 제공합니다. 이 Wrapper를 사용하면 Unity3D 환경에 RLlib의 분산 강화학습 알고리즘을 손쉽게 적용할 수 있습니다.

## 문제상황: 사용자 지정 포트에서 분산 학습 실행

프로젝트의 초기 단계에서 여러 가지 가능성을 빠른 시간 안에 검토하기 위해 단일 컴퓨팅 자원에서 여러가지 실험을 동시에 실행할 필요가 있었습니다.
그리고 이를 위해서는 각 실험에는 개별적인 포트가 할당되어야 했습니다.
ML-Agents에서는 각 환경에서 사용할 포트를 선택할 수 있도록 옵션을 제공하며, 이 기능은 RLlib의 Unity3D Wrapper에서도 사용할 수 있도록 구현되어 있습니다.
그러나 Wrapper에서 이 옵션을 사용하여 포트를 지정하면 실험이 정상적으로 시작되지 않는 문제가 발견되었습니다.
당시 RLlib Wrapper의 문제점은 다음과 같았습니다.

```python

    _BASE_PORT = 5004 # 전역 변수 선언

    ...

    def __init__(self,
                 file_name: str = None,
                 port: Optional[int] = None, # 사용자 지정 포트.
                 seed: int = 0,
                 no_graphics: bool = False,
                 timeout_wait: int = 300,
                 episode_horizon: int = 1000):

    ...

      port_ = port or self._BASE_PORT # 사용자가 포트를 지정한다면 전역 변수를 쓰지 않습니다.
      self._BASE_PORT += 1 # 다음 환경을 위해 전역 변수의 값을 증가시킵니다.
      try:
          self.unity_env = UnityEnvironment(
              file_name=file_name,
              worker_id=0,
              base_port=port_, # 사용자가 포트를 지정한다면 모든 환경이 똑같은 포트를 할당 받습니다.
              seed=seed,
              no_graphics=no_graphics,
              timeout_wait=timeout_wait,
          )

```

사용자 지정이 아닌 기본 (Default) 설정을 사용하면 RLlib의 Unity3D Wrapper가 자동적으로 분산 학습에서 사용하는 각 환경들의 포트를 정의하게 됩니다.
허나, 사용자가 Wrapper를 통해 포트를 지정하면 분산환경에서 사용하는 모든 환경들이 똑같은 포트를 배정받는 문제가 있었습니다.
따라서 첫번째 실행되는 환경만 정상적으로 동작하며 그 외의 환경들은 그 포트를 사용할 수 있을 때까지 실행되지 못하고 기다리게 됩니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-20-open-source-contributions-rllib/before_pr.gif" alt="before PR" width="80%">
  <figcaption style="text-align: center;">[그림2] 문제상황: 지정한 4개의 환경 중 첫 번째로 실행되는 환경만 정상동작</figcaption>
</p>
</figure>

## 문제 해결과정

Ml-Agents 환경은 base_port 외에 worker_id도 받을 수 있고 내부적으로 `base_port + worker_id` 포트에 연결합니다. 따라서, 발견했던 문제를 해결하기 위해 기본 포트를 고정하고 대신 환경의 worker_id를 증가했습니다.

```python
    # Ml-Agents 내부적인 기본 포트를 마추기
    _BASE_PORT_EDITOR = 5004 
    _BASE_PORT_ENVIRONMENT = 5005
    _WORKER_ID = 0

    ...
    
    def __init__(self,
                 file_name: str = None, # 사용자 지정의 컴파일 된 앱. 없으면 Unity3D 에디터에 연결.
                 port: Optional[int] = None, # 사용자 지정 포트.
                 seed: int = 0,
                 no_graphics: bool = False,
                 timeout_wait: int = 300,
                 episode_horizon: int = 1000):

    ...
      # 우선순위: 사용자가 지정한 포트 -> 컴파일 된 앱 -> Unity3D 에디터
      port_ = port or (self._BASE_PORT_ENVIRONMENT
                        if file_name else self._BASE_PORT_EDITOR)
      # 에디터에 연결하면 동시에 한 환경만 사용 가능합니다.
      worker_id_ = Unity3DEnv._WORKER_ID if file_name else 0
      # 포트 대신 worker_id 증가 (Ml-Agents 안에 base_port + worker_id 포트에 자동적으로 연결합니다).
      Unity3DEnv._WORKER_ID += 1
      try:
          self.unity_env = UnityEnvironment(
              file_name=file_name,
              worker_id=worker_id_,
              base_port=port_,
              seed=seed,
              no_graphics=no_graphics,
              timeout_wait=timeout_wait,
          )
```

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-20-open-source-contributions-rllib/after_pr.gif" alt="after PR" width="80%">
  <figcaption style="text-align: center;">[그림3] 문제 해결 후 지정한 모든 환경이 정상동작</figcaption>
</p>
</figure>
로컬환경에서 문제를 해결하는 것은 크게 어렵지 않았지만, 동일한 문제를 겪고 있을 다른 사람들을 위해 Pull Request (이하 PR)를 통해 RLlib에 기여하기로 결정했습니다.
문제해결(Bug Fix) PR을 제출할 때는 해당 문제가 재발하지 않도록 적절한 유닛테스트를 추가하는 것이 매우 중요합니다.
더불어 유닛테스트를 통해 다른 유지관리자(Maintainer)들이 변경 사항을 좀 더 평가하기 쉽게하는 효과를 기대할 수 있습니다.
다음과 같이 문제 해결과 더불어 사용자가 직접 포트를 지정하는 경우에도 모든 환경들이 각각 다른 포트를 할당받는지 확인하는 유닛테스트를 추가했습니다.

```python

import unittest
from unittest.mock import patch

from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv


@patch("mlagents_envs.environment.UnityEnvironment")
class TestUnity3DEnv(unittest.TestCase):
    def test_port_editor(self, mock_unity3d):
        """Test if the environment uses the editor port
         when no environment file is provided"""

        _ = Unity3DEnv(port=None)
        args, kwargs = mock_unity3d.call_args
        mock_unity3d.assert_called_once()
        self.assertEqual(5004, kwargs.get("base_port"))

    def test_port_app(self, mock_unity3d):
        """Test if the environment uses the correct port
        when the environment file is provided"""

        _ = Unity3DEnv(file_name="app", port=None)
        args, kwargs = mock_unity3d.call_args
        mock_unity3d.assert_called_once()
        self.assertEqual(5005, kwargs.get("base_port"))

    def test_ports_multi_app(self, mock_unity3d):
        """Test if the base_port + worker_id
        is different for each environment"""

        _ = Unity3DEnv(file_name="app", port=None)
        args, kwargs_first = mock_unity3d.call_args
        _ = Unity3DEnv(file_name="app", port=None)
        args, kwargs_second = mock_unity3d.call_args
        self.assertNotEqual(
            kwargs_first.get("base_port") + kwargs_first.get("worker_id"),
            kwargs_second.get("base_port") + kwargs_second.get("worker_id"))

    def test_custom_port_app(self, mock_unity3d):
        """Test if the base_port + worker_id is different
        for each environment when using custom ports"""

        _ = Unity3DEnv(file_name="app", port=5010)
        args, kwargs_first = mock_unity3d.call_args
        _ = Unity3DEnv(file_name="app", port=5010)
        args, kwargs_second = mock_unity3d.call_args
        self.assertNotEqual(
            kwargs_first.get("base_port") + kwargs_first.get("worker_id"),
            kwargs_second.get("base_port") + kwargs_second.get("worker_id"))
```

PR을 제출하면 유지관리자와의 커뮤니케이션 프로세스를 통해 변경사항에 대해 질의를 주고받거나 필요시 추가적인 작업을 수행하기도 합니다.
이 프로세스를 통해 새로운 변경 사항이 프로젝트의 지침을 잘 준수하는지 확인할 수 있기 때문에 유지관리자와의 소통과정은 오픈소스 컨트리뷰션에서 가장 중요한 부분 중 하나입니다.

다음과 같이 담당자가 추가적으로 Continuous Integration (CI) 스크립트 및 라이브러리 requirements을 갱신해달라는 요청을 했습니다. 요청들을  처리한 후, 해당 [PR](https://github.com/ray-project/ray/pull/13519)은 머지되어 Ray 1.3 버전부터 포함되었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-20-open-source-contributions-rllib/communication.png" alt="after PR" width="80%">
  <figcaption style="text-align: center;">[그림4] 담당자의 요청 처리하는 과정</figcaption>
</p>
</figure>

# 맺음말

오픈소스를 사용하다 보면 언제든지 문제상황에 봉착할 수 있습니다.
그리고 이때 누구나 문제에 대한 해결책을 제안할 수 있습니다.
오픈소스의 장점은 자유로운 기여를 통해 누구나 커뮤니티에 더 나은 도구를 제공할 수 있다는 것입니다.

***That is the beauty of contributing to Open-Source***

*A Special Thanks to [Jinwoo Park](https://github.com/Curt-Park) for helping with this post.*
