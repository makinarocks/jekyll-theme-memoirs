---
layout: post
title: 마키나락스에서의 Kubeflow-Katib 의 활용과 컨트리뷰션 공유
author: jaeyeon kim
categories: [open_source, kubeflow]
image: assets/images/2021-08-31-open-source-contributions-katib/hpo-result.png
---

[마키나락스](http://www.makinarocks.ai/)의 Platform 팀에서는 [kubeflow](https://github.com/kubeflow/kubeflow)를 기반으로 하여 ML/DL 모델의 실험과 배포의 간극을 줄이는 MLOps 플랫폼을 개발하고 있습니다.
kubeflow 는 ML Workflow 를 kubernetes-native 하게 실행하고 관리할 수 있는 플랫폼이지만, 아직 v1.0 이 released 된 지 약 1년 반 정도밖에 지나지 않은 프로젝트이기에 kubernetes 에 익숙하지 않은 Data Scientist, ML Engineer 가 사용하기에는 부족한 점이 다수 존재합니다.

따라서 저희 Platform 팀에서는 kubeflow 의 여러 구성요소들을 활용하되, 특정 기능은 우회해서 사용하고, 또 특정 기능은 자체적으로 구현하여 적절히 커스터마이징하여 사용하고 있습니다. 또한 그중 kubeflow 프로젝트에 보편적으로 반영할만한 기능들은 PR 을 직접 날려 컨트리뷰션을 진행하고 있습니다.

본 포스팅에서는 kubeflow 의 구성요소 중 **Hyperparameter Tuning and Neural Architecture Search** 기능을 제공하는 [katib](https://github.com/kubeflow/katib)를 사용하며 겪었던 불편 사항과 해당 기능을 구현하여 오픈소스 프로젝트에 기여한 경험을 공유드리려고 합니다. katib 에 대해 보다 자세한 정보가 궁금하시다면 다음 [공식 문서](https://www.kubeflow.org/docs/components/katib/overview/)를 확인해 주시기 바랍니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-31-open-source-contributions-katib/hpo-result.png" alt="" width="100%">
  <figcaption style="text-align: center;">[그림 1] Simple HPO with katib</figcaption>
</p>
</figure>

# 컨트리뷰션 배경

## Katib

마키나락스에서는 kubeflow 와 katib 를 통합하여, 각 팀원의 로컬 머신 내에서 프로세스 단위의 parallelized hyperparmeter optimization(이하 Hpo) 가 아닌, 약 10 개의 gpu 서버를 통합한 사내 kubernetes cluster 안에서 pod 단위의 parallelized Hpo 을 수행하고 있습니다.

Hpo 의 성능을 끌어올리는 가장 흔한 방법으로는 대규모의 hyperparameter search space에 대해 많은 양의 컴퓨팅을 수행하는 방법을 사용하기에, 고스펙의 서버와 많은 시간을 필요로 합니다.
따라서 흔히 사용하는 Hpo package 인 [Scikit-Optimize](https://scikit-optimize.github.io/stable/user_guide.html), [Hyperopt](https://github.com/hyperopt/hyperopt), [Optuna](https://optuna.readthedocs.io/en/stable/)를 로컬 머신의 제한된 스펙 내에서 수행하는 경우 상당한 시간 소모가 불가피합니다. 하지만 kubernetes 와 같은 container orchestration system 을 활용해 node 단위의 병렬 작업과 리소스 관리 최적화 및 스케줄링을 자동화한다면 훨씬 효율적인 실험을 수행할 수 있습니다.

물론 Optuna 와 Ray-tune 을 비롯한 Hpo package 들에서도, 다수의 서버를 cluster 로 구성하거나, 혹은 기존 kubernetes cluster 에서 worker 를 나누어 병렬 작업을 수행할 수 있는 기능을 지원하고 있습니다. 하지만 이들은 처음부터 kubernetes native 하게 설계되지 않은 프로젝트이기 때문에 kubernetes cluster에서 사용하기에는 다소 활용도가 떨어지는 부분이 존재합니다.

katib 는 이러한 문제를 효과적으로 해결하기 위해 시작된 프로젝트입니다.
katib 는 자체적으로 hyperparameter search algorithm 을 구현하기도 하지만, 가장 큰 특징은 skopt, hyperopt, chocolate, optuna 와 같은 외부 라이브러리들의 기존 method 들을 **쉽게 통합할 수 있는 인터페이스를 제공**한다는 점입니다. <br>
인터페이스에 맞게 통합할 때 필요한 [가이드](https://github.com/kubeflow/katib/blob/master/docs/new-algorithm-service.md)도 자세히 제공하고 있으며, 실제로 이에 맞게 skopt, chocolate, optuna 등의 구현체가 katib 에 [통합되어 있습니다](https://github.com/kubeflow/katib/tree/master/pkg/suggestion/v1beta1).

즉, katib 는 hyperparameter search algorithm 의 구현체들을 모아둔 프로젝트라기보다는, 구현체들이 kubernetes 내의 자원들을 효율적으로 사용하며 Hpo 를 수행할 수 있도록 API 서비스화를 담당하며, 서비스를 안정적으로 제공할 수 있는 인프라를 제공하는 프로젝트에 가깝다고 표현할 수 있습니다. <br>
kubernetes 에 익숙하신 독자분들이라면 kubernetes 에서는 CRI, CNI, CSI 만을 지원하여, 다양한 container-runtime, network, storage vendor 를 plugin 의 형태로 통합하는 컨셉과 유사하다고 보실 수도 있습니다.

하지만 아직 katib 는 이제 막 **v0.12.0** release 를 앞두고 있는 한창 발전해나가는 프로젝트이기에, 마키나락스에서는 katib 를 사용하며 여러 가지 이슈를 맞닥뜨리게 되었습니다.


## 이슈 상황

katib 는 Hpo 한 세트를 `Experiment` 라는 [custom resource](https://kubernetes.io/ko/docs/concepts/extend-kubernetes/api-extension/custom-resources/) 로 정의하고 관리하고 있습니다. <br>
따라서 kubernetes 의 다른 custom resource 관리 방식과 동일하게 사용자가 `Experiment` 를 생성하기 위해서는 다음과 같은 형태의 yaml file 혹은 json 을 만들어 kubernetes api server 로 생성 요청을 보내야 합니다.
하지만, katib 에서 내부적으로 정해놓은 규칙을 지키지 않은 형태로 `Experiment` 생성 요청을 수행할 경우, 실제로는 `Experiment` 가 정상적으로 생성되지 않았음에도 불구하고 사용자가 보기에는 해당 `Experiment` 의 상태가 `Running` 혹은 `Creating` 으로 보여 **제대로 동작하는 것으로 착각**하게 되는 문제가 자주 발생하였습니다.

그 중 저희가 자주 겪었던 문제 상황의 예를 들면 다음과 같습니다. <br>
- 1) `Experiment` 의 이름을 정해진 rule 에 어긋나게 생성한 [경우](https://github.com/kubeflow/katib/issues/1538)
- 2) `Experiment` 의 `suggestion algorithm` 관련 필드를 잘못 입력한 [경우](https://github.com/kubeflow/katib/issues/1126)
- 3) `Experiment` 의 `primary-container` 필드를 잘못 입력한 [경우](https://github.com/kubeflow/katib/issues/1542)
- 4) `Experiment` 생성 시 `sidecar.istio.io/inject: "true"` 를 명시하지 않은 [경우](https://github.com/kubeflow/katib/issues/1412)

kubernetes 와 kubeflow, katib 에 익숙한 사용자라면 여러 구성 요소들의 log 를 일일이 확인해보면서 그 원인을 파악하고 해결할 수 있지만, 익숙하지 않은 사용자들에게는 문제를 해결하기 매우 어려운 환경이었습니다. <br>
사용자의 실수에 대한 처리를 강하게 하지 않아서 생기는 문제, [Fail-fast](https://en.wikipedia.org/wiki/Fail-fast)를 고려하지 않아 생긴 문제를 다수 확인할 수 있었습니다.


# 컨트리뷰션 진행

마키나락스에서도 katib 을 사용하며 비슷한 문제들을 반복해서 겪게 되었고, 단순히 katib 를 그대로 사용하는 방식으로는 kubernetes 에 익숙하지 않은 Data Scientist 와 ML Engineer 가 사용하기에는 어렵다는 결론을 내리게 되었습니다.

따라서 1차적으로는 사용자가 직접 katib `Experiment` 의 yaml 을 모두 작성하는 형태가 아니라, 소수의 인터페이스만 제공하여 안전한 사용을 할 수 있도록 하였습니다. 사용자가 `Experiment` 에 필요한 필수적인 정보만 입력하면 `Experiment` 의 생성을 위한 yaml 파일의 대부분을 채워주는 **CLI tool 을 제공**하여, 안전한 사용과 더불어 katib 와 yaml 문법에 익숙하지 않은 사용자도 쉽게 접근할 수 있도록 하였습니다.

또한, 2차적으로는 katib 레이어에서도 비정상적인 요청에 대한 케이스를 조금 더 정교하게 처리하기 위해서 직접 katib 프로젝트에 컨트리뷰션을 진행하였습니다. 이 때, katib Maintainer 들이 문제 상황을 한 눈에 파악할 수 있도록 해당 이슈를 재현할 수 있는 명령과 스크린샷을 자세하게 첨부한 issue 를 우선 생성하였으며, 이후 해당 기능을 추가한 코드와 테스트 코드를 추가한 **PR**을 추가하는 순서로 진행하였습니다.

이렇게 머지된 두 가지 PR 에 대한 내용을 소개하겠습니다.


### Experiment Naming

[첫 번째](https://github.com/kubeflow/katib/pull/1541)로는 `Experiment` 의 naming convention 이 정해져 있었지만, 이를 체크하는 로직이 너무 뒤에 있어서 사용자의 입장에서는 생성 요청한 experiment 가 `CREATED` 으로 영원히 멈춰있는 이슈였습니다.

우선 정확한 로그를 첨부한 [issue](https://github.com/kubeflow/katib/issues/1538) 를 생성하여, 해당 상황이 발생하지 않도록 kubernetes 의 [ValidatingAdmissionWebhook](https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/#validatingadmissionwebhook) 에서 block 처리를 하거나, 혹은 최소한 해당 상황이 발생하면 `Experiment` 의 status 를 `FAILED` 로 변경하도록 수정하는 것이 어떻겠냐는 구현 방향을 함께 전달하였습니다.

이후 katib Maintainer 로부터 해당 이슈는 버그가 맞고, Fail-Fast 를 위해서 `ValidatingAdmissionWebhook` 에서 check 하는 것이 좋겠다고 동의해 주었습니다. <br>
이후에 제가 직접 구현하고 싶다는 의사를 전달하여 구현을 시작하였고, 아래와 같이 간단한 정규 표현식을 사용해 naming convention 을 검증하는 코드와 테스트 코드를 추가한 PR 을 생성하여 Maintainer 의 review 를 거쳐 머지되었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-31-open-source-contributions-katib/1-code.png" alt="" width="90%">
  <figcaption style="text-align: center;">[그림 2] Changes in PR</figcaption>
</p>
</figure>


### Hyperparameter Search Algorithm Setting

[두 번째](https://github.com/kubeflow/katib/pull/1600)로는 `Experiment` 에서 Hyperparameter Search Algorithm 을 선택할 때, 해당 algorithm 의 세팅을 잘못 입력한 경우에 대한 Validation 로직이 없는 algorithm 으로 생성 요청한 경우, experiment 가 `RUNNING` 으로 영원히 멈춰있는 이슈였습니다. katib 에서는 해당 [issue](https://github.com/kubeflow/katib/issues/1126) 를 이미 파악하고 있었지만, 약 1 년째 진행되지 않고 open & frozen 상태로 남아있는 상황이었습니다.

마키나락스에서도 bayesian optimization algorithm 으로 hyperparameter search 를 수행하는 실험을 진행하던 중 동일한 문제 상황을 겪었기에, 프로젝트에 직접 기여하기로 결정하였습니다.

먼저 katib 프로젝트의 해당 이슈에 이슈에 대해 조금 더 자세한 안내를 요청하는 코멘트를 남겼고, katib 의 Maintainer 중 한 명인 [@andreyvelich](https://github.com/andreyvelich)가 친절하고 빠르게 답변해주어 소스코드 상의 어디가 문제인지, 어떻게 고쳐야 하는지에 대해 보다 빠르게 파악할 수 있었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-31-open-source-contributions-katib/katib-hp-search-issue-comment.png" alt="" width="80%">
  <figcaption style="text-align: center;">[그림 3] Maintainer's guidance from issue comments</figcaption>
</p>
</figure>

요약하자면 katib 에서 제공하는 hyperparameter search algorithm 중 Hyperopt 는 [다음](https://github.com/kubeflow/katib/blob/3fadef637ad17458f629a4baeba7fd38205a1510/pkg/suggestion/v1beta1/hyperopt/service.py#L57)과 같이 algorithm 의 Validation 과정을 제공하고 있었지만, skopt 를 비롯한 일부 algorithm 은 Validation 과정을 제공하고 있지 않는다는 문제였습니다.

따라서 이를 해결하기 위해 우선 skopt 의 optimizer 의 공식 api [문서](https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html)를 보며 skopt 에서 필요한 validation 이 무엇인지 확인하였습니다. 이후 정해진 validation 을 수행하는 로직을 추가하고, 각각의 경우에 대한 테스트케이스를 검증하는 test code 를 추가한 PR 을 작성하였습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-31-open-source-contributions-katib/validation-1.png" alt="" width="100%">
</p>
<p align="center">
  <img src="/assets/images/2021-08-31-open-source-contributions-katib/validation-2.png" alt="" width="100%">
  <figcaption style="text-align: center;">[그림 4] Added validation code</figcaption>
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-31-open-source-contributions-katib/skopt-test-code.png" alt="" width="85%">
  <figcaption style="text-align: center;">[그림 5] Test code for each test case</figcaption>
</p>
</figure>

이후 다음과 같이 Maintainer 로부터 katib 내의 convention 을 통일하는 등의 리뷰를 거쳐 머지되었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-31-open-source-contributions-katib/2-review-comments.png" alt="" width="85%">
  <figcaption style="text-align: center;">[그림 6] Maintainer's guidance from review comments</figcaption>
</p>
</figure>


# 맺음말

위의 PR 외에도 kubeflow/katib 를 비롯해 [kubeflow/kubeflow](https://github.com/kubeflow/kubeflow), [kubeflow/pipeline](https://github.com/kubeflow/pipelines) 등의 kubeflow 관련 프로젝트들을 마키나락스에서 활용하며 만났던 이슈들과 그 해결책들을 지속적으로 제시하다 보니, 기존 kubeflow member 의 추천을 받아 kubeflow organization 의 member 로 [합류](https://github.com/kubeflow/internal-acls/pull/487)하게 되었습니다. 

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-08-31-open-source-contributions-katib/join-kubeflow.png" alt="" width="100%">
  <figcaption style="text-align: center;">[그림 7] Joined Kubeflow Organization</figcaption>
</p>
</figure>

마키나락스에서는 kubeflow 외에도 다양한 오픈소스들을 적극적으로 활용하고 있으며, 단순한 사용자에 그치기보다는 오픈소스로부터 받았던 도움을 다시 커뮤니티에 돌려주는 컨트리뷰터가 되기를 장려하고 있습니다. 저 또한 kubeflow member 로써 앞으로 더 적극적인 기여를 지속하는 것을 목표로 하고 있습니다.

오픈소스를 사용하다 보면 언제든지 문제상황에 봉착할 수 있습니다.
그리고 이때 누구나 문제에 대한 해결책을 제안할 수 있습니다.
오픈소스의 장점은 자유로운 기여를 통해 누구나 커뮤니티에 더 나은 도구를 제공할 수 있다는 것입니다.

***That is the beauty of contributing to Open-Source***
