# RE:Wheeler - Reinforcement Learning experimental framework ( Policy Gradients )
* Project is focused on [HER](https://openai.com/blog/ingredients-for-robotics-research/) and trying to address [request for research from OpenAI](https://openai.com/blog/ingredients-for-robotics-research/#requestsforresearchheredition) on this topic.
  - HER + multi-step returns : Floating-N-steps + HER + GAE
  - On-policy HER : Intercorporating PPO algorithm ( On-Policy and Off-Policy settings )
  - Combine HER with recent advances in RL : combining HER with cooperation of learning algorithm in one shot (DDPG + PPO; or other combination of algorithm is possible)
  - Richer value functions : not actually what was asked, however MROCS approach givin richer signal for learning ~ theoretically ~
  - Faster information propagation : possible to experiment with sync_delta + tau of soft-actor-critic for cooperation algorithm, and different techniques of sync

More Information
===
check blog [here](https://rezer0dai.github.io/rewheeler/).

### Environment
  - state_size=33, action_size=4 as default UnityML - Reacher environment provides
  - 20 arms environment, used shared actor + critic for controlling all arms
  - How to install :
  - environment itself : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
    - unpack package to ./reach/ folder inside this project

  - replicate results by running [notebooks](https://rezer0dai.github.io/rewheeler/#jupyter-notebooks)
