---
layout: post
title: Open Source Contributions - RLlib
author: yuri rocha
categories: [open-source]
image: assets/images/2021-08-10-open-source-contributions-rllib/balance.png
---

[마키나락스](http://www.makinarocks.ai/)의 OLP(Off-line Programming) 팀에서는 산업군에서 사용하는 Multi-Robot Arm의 경로계획(Path Planning)* 문제를 강화학습을 이용하여 풀고 있습니다.
**경로계획이란 다수의 로봇팔들이 효과적으로 동작할 수 있는 경로를 생산하는 문제입니다.*
실재 로봇이 작용하면 시간 및 안전 문제 생길 수 있기 때문에 강화학습으로 경로계획 풀기 위해 시뮬레이터가 필요합니다.
[지난 블로그포스팅](https://makinarocks.github.io/Building-a-Reinforcement-Learning-Environment/)에서는 저희 시뮬레이터 개선 과정을 공유드렸습니다.

본 포스팅에서는 오픈소스 라이브러리(RLlib)를 사용하며 발견한 문제의 원인 분석부터 컨트리뷰션을 통한 문제해결까지의 과정을 공유드리려고 합니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-10-open-source-contributions-rllib/OLP_example_image.jpg" alt="OLP" width="80%">
  <figcaption style="text-align: center;">[그림1] Multi-Robot Arm 실재 환경</figcaption>
</p>
</figure>

# 오픈소스 컨트리뷰션

오픈 소스는 허가된 라이센스로 보고, 사용하고, 수정하고, 배포하기 위해 대중이 사용할 수 있는 소스 코드입니다.
오픈 소스 프로젝트의 핵심은 접근성이 뛰어난 가치 있는 오픈 소스 소프트웨어를 만드는 것이지만, 이러한 프로젝트에 기여하는 것은 컨트리뷰터에게 도움이 될 수 있습니다.
사용하는 소프트웨어를 유지 관리하는 동시에 기술적인 능력도 연마할 수 있을 뿐더러 오픈소스 커뮤니티에 기여할 수 있기 때문입니다.

# 컨트리뷰션 배경

## ML-Agents


ML-Agents는 Unity3D 게임 엔진을 기반으로 작성된 시뮬레이터로서, 지능형 에이전트의 학습 환경으로 사용되는 오픈소스 프로젝트입니다.
[지난 블로그 포스팅](https://makinarocks.github.io/Building-a-Reinforcement-Learning-Environment/)에서 OLP 프로젝트를 위해 정의한 Unity3D 환경에 대해 소개해 드린 적이 있습니다.

## Ray RLlib

Ray는 파이썬 기반의 분산처리 프레임워크로, 다양한 기계학습 라이브러리를 포함하고 있습니다. 그 중 RLlib은 확장 가능한 (Scalable) 분산 강화학습 환경을 간편히 구축할 수 있도록 도와주는 라이브러리입니다. 
OLP팀에서는 RLlib을 사용하여 대규모의 강화학습을 수행하고 있습니다.
RLlib은 기본적으로 Unity3D의 ML-Agents 환경에 연결할 수 있는 [Wrapper](https://medium.com/distributed-computing-with-ray/reinforcement-learning-with-rllib-in-the-unity-game-engine-1a98080a7c0d)을 제공합니다. 이 Wrapper를 사용하면 Unity3D 환경에 RLlib의 분산 강화학습 알고리즘을 손쉽게 적용할 수 있습니다.

## 문제상황: 사용자 지정 포트에서 분산 학습 실행

프로젝트의 초기 단계에서는 여러 가지 가능성을 빠른 시간 안에 검토할 필요가 있었습니다.
이때 주어진 컴퓨팅 자원을 최대한 효율적으로 사용하기 위해 동일 서버에서 여러가지 실험을 동시에 실행할 필요가 있었습니다. 그리고 이를 위해 각 실험에는 개별적인 포트가 할당되어야 했습니다.
ML-Agents에서는 각 환경에서 사용할 포트를 선택할 수 있도록 옵션을 제공합니다.
그리고 이 기능은 RLlib의 Unity3D Wrapper에서도 사용할 수 있도록 구현되어 있습니다.
그러나 Wrapper에서 이 옵션을 사용하여 포트를 지정하면 실험이 정상적으로 시작되지 않는 현상을 확인했습니다.
분석한 RLlib 구현상의 문제점은 다음과 같았습니다.

```python

    _BASE_PORT = 5004

    ...

    def __init__(self,
                 file_name: str = None,
                 port: Optional[int] = None,
                 seed: int = 0,
                 no_graphics: bool = False,
                 timeout_wait: int = 300,
                 episode_horizon: int = 1000):

    ...

      port_ = port or self._BASE_PORT
      self._BASE_PORT += 1
      try:
          self.unity_env = UnityEnvironment(
              file_name=file_name,
              worker_id=0,
              base_port=port_,
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
  <img src="/assets/images/2021-08-10-open-source-contributions-rllib/before_pr.gif" alt="before PR" width="80%">
  <figcaption style="text-align: center;">[그림2] 문제상황: 지정한 4개의 환경 중 첫 번째로 실행되는 환경만 정상동작</figcaption>
</p>
</figure>

## 문제 해결과정

로컬환경에서 문제를 해결하는 것은 크게 어렵지 않았지만, 동일한 문제를 겪고 있을 다른 사람들을 위해 Pull Request (이하 PR)를 통해 RLlib에 기여하기로 결정했습니다.

```python
    # Default base port when connecting directly to the Editor
    _BASE_PORT_EDITOR = 5004
    # Default base port when connecting to a compiled environment
    _BASE_PORT_ENVIRONMENT = 5005
    # The worker_id for each environment instance
    _WORKER_ID = 0

    ...
    
    def __init__(self,
                 file_name: str = None,
                 port: Optional[int] = None,
                 seed: int = 0,
                 no_graphics: bool = False,
                 timeout_wait: int = 300,
                 episode_horizon: int = 1000):

    ...

      port_ = port or (self._BASE_PORT_ENVIRONMENT
                        if file_name else self._BASE_PORT_EDITOR)
      # cache the worker_id and
      # increase it for the next environment
      worker_id_ = Unity3DEnv._WORKER_ID if file_name else 0
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
  <img src="/assets/images/2021-08-10-open-source-contributions-rllib/after_pr.gif" alt="after PR" width="80%">
  <figcaption style="text-align: center;">[그림3] 문제 해결 후 지정한 모든 환경이 정상동작</figcaption>
</p>
</figure>

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

다음과 같이 담당자의 추가적인 요청을 처리한 후, 해당 [PR](https://github.com/ray-project/ray/pull/13519)은 머지되어 Ray 1.3 버전부터 포함되었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-10-open-source-contributions-rllib/communication.png" alt="after PR" width="80%">
  <figcaption style="text-align: center;">[그림4] 담당자의 요청 처리하는 과정</figcaption>
</p>
</figure>

# 맺음말

오픈 소스 프로젝트를 사용하다 보면 언제든지 문제상황에 봉착할 수 있습니다.
그리고 이때 누구나 이러한 문제에 대한 해결책을 제안할 수 있습니다.
오픈 소스의 장점은 자유로운 기여를 통해 누구나 커뮤니티에 더 나은 도구를 제공할 수 있다는 것입니다.

***That is the beauty of contributing to Open-Source***
