---
layout: post
title: Open Source Contributions - rllib
author: yuri rocha
categories: [open-source]
image: assets/images/2021-08-10-open-source-contributions-rllib/balance.png
---

[마키나락스](http://www.makinarocks.ai/)의 OLP(Off-line Programming) 팀에서는 시뮬레이터 상에서 실제 봇이 사용할 robot program을 생성하는 Offline Programming 문제의 자동화를 연구 개발하고 있습니다. Reinforcement Learning, Combinatorial Optimization, 각종 robotics planning 알고리즘들을 사용하여 문제를 풀고 있습니다. [지난 블로그포스팅](https://makinarocks.github.io/Building-a-Reinforcement-Learning-Environment/)에서는 저희 시뮬레이터 개선 과정을 공유했습니다.

본 블로그에서는 프로젝트를 하다가 하게 된 오픈 소스 컨트리뷰션에 대해서 다뤄보도록 하겠습니다.

# 오픈 소스 컨트리뷰션이란...

오픈 소스는 허가된 라이센스로 보고, 사용하고, 수정하고, 배포하기 위해 대중이 사용할 수 있는 소스 코드입니다. 오픈 소스 프로젝트의 핵심은 접근성이 뛰어난 가치 있는 오픈 소스 소프트웨어를 만드는 것이지만, 이러한 프로젝트에 기여하는 것은 컨트리뷰터에게 도움이 될 수 있습니다. 사용하는 소프트웨어를 유지 관리하는 동시에 기술적인 능력도 연마할 수 있기 때문입니다. 마키나락스에서는 근무 시간에 오픈 소스에 컨트리뷰션을 할 수 있을 뿐만 아니라 이를 권장하고 있습니다.

# 컨트리뷰션 사유

## Mlagents

Mlagents는 Unity 3D 게임 엔진을 기반으로 작성된 시뮬레이션을 지능형 에이전트의 학습 환경으로 사용하는 오픈 소스 프로젝트입니다. [지난 블로그포스팅](https://makinarocks.github.io/Building-a-Reinforcement-Learning-Environment/)에서 저희의 Unity3D에 기반 맞춤형 환경을 보여줬습니다.

## Ray Rllib

Rllib은 파이썬 기반의 분산 기계학습 프레임워크인 Ray상에서 동작하는 강화학습 라이브라리입니다. OLP팀에서는 Rllib을 사용하여 대규모 강화학습을 수행합니다. Rllib은 기본적으로 Unity3D mlagents 환경에 연결할 수 있는 wrapper을 제공합니다.

## 사용자 지정 포트에서 분산 학습 실행

프로젝트의 초기 단계에서 여러 가지 접근 방식을 동시에 테스트해야 했습니다. 이 실험은 동일한 서버에서 실행되었으므로 각 실험에서 사용할 포트를 신중하게 선택해야 했습니다. Mlagents를 통해 환경에 연결할 때 사용할 원하는 포트를 선택할 수 있습니다. 이 기능은 Rllib의 unity3D env wrapper에 포함되어 있습니다. 그러나 실험을 수행할 때 지정한 포트를 사용하면 실험이 시작되지 않는다는 것을 알게 되었습니다. Rllib 코드를 살펴봐서 버그를 발견했습니다. 기본 포트를 사용하면 rllib의 unity3d env wrapper가 자동적으료 분산 학습 쓸 각 환경들에게 다른 포트를 줍니다. 그러나, wrapper에게 지정 포트를 주면 모든 환경들이 똑같은 포트를 받습니다. 따라서 첫번쩨 환경만 성공적으로 시작하며 남은 환경들이 그 포트를 사용할 수 있을 때까지 기다리고 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-10-open-source-contributions-rllib/before_pr.gif" alt="before PR" width="80%">
  <figcaption style="text-align: center;">[그림1] - 버그 고치기 전</figcaption>
</p>
</figure>

## 버그 고치기

본 버그를 고치는 데 몇 분밖에 안 걸렸습니다. 그러나 수정된 코드은 로컬 컴퓨터에서만 작동하여 수정된 코드가 있는 PR를 하기로 했습니다.

버그 수정 PR을 제출할 때 현재 상태에서는 실패하지만 버그 수정이 도입된 후 통과되는 우닛테스트를 추가하는 것이 매우 중요합니다. 유닛테스트를 추가하면 유지관리자가 변경 사항을 평가하기 더 쉽고 버그가 다시 생기는 추가 변경 사항을 방지할 수 있기 때문입니다. 따라서 추가적으로 사용자 지정 포트를 써도 모든 환경들이 다른 포트를 받고 있는지 확인하는 유닛테스트도 추가했습니다.

PR 제출 후, 유지관리자가 코드에 대해 질문하거나 변경 또는 새로운 추가를 요청할 수 있는 커뮤니케이션 프로세스가 시작됩니다. 이 프로세스는 오픈 소스에 컨트리뷰션할 때 가장 중요한 부분 중 하나입니다. 이 프로세스를 통해 코드 베이스와 새로운 변경 사항을 잘 통합하고 컨트리뷰션이 리포지토리 지침을 준수하는지 확인할 수 있기 떼문입니다.

담당자의 요청을 처리한 후, PR이 머지되어 Ray 1.3 버전부터 포함되었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-10-open-source-contributions-rllib/after_pr.gif" alt="after PR" width="80%">
  <figcaption style="text-align: center;">[그림1] - 버그 고치기 후</figcaption>
</p>
</figure>

# Parting Words

오픈 소스 코드를 사용하면 언제든지 버그를 찾을 수 있습니다. 마찬가지로 누구나 이러한 버그에 대한 해결책을 제안할 수 있습니다. 오픈 소스의 장점은 누구나 커뮤니티에게 더 나은 도구들를 제공할 수 있다는 것입니다.
