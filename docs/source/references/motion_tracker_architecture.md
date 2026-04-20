# Motion Tracker / Universal-Token 架构深挖

## 1. 这份文档在回答什么

这份文档聚焦的是 **SONIC 训练/部署代码里的 motion tracker 控制器本体**，也就是 `UniversalTokenModule` 这一套

`encoder -> FSQ -> decoder`

链路，而不是 PICO/VR 硬件本身的接入文档。

用户最关心的几个问题，在这份代码里可以落到下面这几个结论：

1. 真正的主干模型是 `gear_sonic/trl/modules/universal_token_modules.py` 里的 `UniversalTokenModule`。
2. 每个 encoder / decoder 本质上都不是独立手写网络，而是由 `gear_sonic/trl/modules/base_module.py` 动态构造出来的 **MLP**。
3. 对发布配置 `sonic_release` 而言，`g1` 这一路的真实生效链路是：

   `g1 tokenizer obs -> g1 encoder -> FSQ(2 x 32 token) -> g1_kin + g1_dyn`

4. `g1_kin` 是训练期辅助解码器，只用于重建和 cycle consistency；真正输出机器人动作的是 `g1_dyn`。
5. FSQ 在这个仓库里 **不是单独预训练**，也 **没有仓库内自定义的 commitment/codebook/EMA loss**；它是随 PPO 主目标和 auxiliary losses 一起端到端训练的。

---

## 2. 代码文件地图

### 2.1 主干模型

| 文件 | 作用 |
| --- | --- |
| `gear_sonic/trl/modules/universal_token_modules.py` | 核心 ATM / motion tracker 模型；负责 encoder 选择、FSQ 量化、decoder 调用、aux loss 输入组织 |
| `gear_sonic/trl/modules/base_module.py` | 所有 encoder / decoder 的基础网络工厂；按配置动态生成 MLP，并处理时间维 flatten / reshape |
| `gear_sonic/trl/modules/actor_critic_modules.py` | PPO actor 包装器；训练时要求 backbone 返回 aux loss，部署时也支持外部 token 直连 decoder |

### 2.2 网络配置

| 文件 | 作用 |
| --- | --- |
| `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml` | 通用 universal-token 组装配置 |
| `gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml` | `g1_encoder` 的默认 MLP 模板 |
| `gear_sonic/config/actor_critic/decoders/g1_kin_mf_mlp.yaml` | `G1 Kinematic decoder` 的默认 MLP 模板 |
| `gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml` | `G1 Dynamic decoder` 的默认 MLP 模板 |
| `gear_sonic/config/actor_critic/quantizers/fsq.yaml` | FSQ 量化器配置入口，目标类是 `vector_quantize_pytorch.FSQ` |

### 2.3 真实实验配置

| 文件 | 作用 |
| --- | --- |
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml` | 发布版训练配置；会覆盖 `all_mlp_v1.yaml` 中一部分输入/输出定义 |
| `gear_sonic/config/algo/ppo_im_phc.yaml` | PPO 超参数 |
| `gear_sonic/config/trainer/trl_ppo_aux.yaml` | 使用带 auxiliary loss 的 trainer |

### 2.4 tokenizer 观测和 encoder 采样

| 文件 | 作用 |
| --- | --- |
| `gear_sonic/config/manager_env/observations/tokenizer/unitoken_all_noz.yaml` | tokenizer observation 组装入口 |
| `gear_sonic/config/manager_env/observations/terms/*.yaml` | 各个 observation term 的绑定 |
| `gear_sonic/envs/manager_env/mdp/observations.py` | tokenizer / policy obs 的真实计算逻辑 |
| `gear_sonic/config/manager_env/commands/terms/motion.yaml` | 跟踪命令配置（future frame 数、dt、motion lib、采样设置） |
| `gear_sonic/envs/manager_env/mdp/commands.py` | `TrackingCommand`；决定 `encoder_index`、future motion、joint targets 等 |

### 2.5 训练损失

| 文件 | 作用 |
| --- | --- |
| `gear_sonic/config/aux_losses/universal_token/g1_recon_and_all_latent.yaml` | 发布版 auxiliary loss 组合和权重 |
| `gear_sonic/config/aux_losses/terms/*.yaml` | 每个 auxiliary loss 的配置 |
| `gear_sonic/trl/losses/token_losses.py` | `g1_recon`、latent alignment、cycle consistency 的实现 |
| `gear_sonic/trl/trainer/ppo_trainer_aux_loss.py` | PPO 总损失中把 aux loss 加回去 |

### 2.6 部署 / 导出

| 文件 | 作用 |
| --- | --- |
| `gear_sonic/utils/inference_helpers.py` | 导出 encoder-only / decoder-only / encoder+decoder ONNX |
| `gear_sonic_deploy/policy/release/observation_config.yaml` | 发布版部署观测配置 |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` | C++ 参考实现，定义了部署侧观测 registry 和维度 |

---

## 3. 主干架构：这套 motion tracker 到底怎么跑

从 `UniversalTokenModule` 的实现看，主干数据流非常清楚：

```text
tokenizer obs
   |
   +--> g1 encoder
   +--> teleop encoder
   +--> smpl encoder
   |
   +--> 根据 encoder_index 选出当前 env 真正使用的 encoder
   |
latent (pre-quantization)
   |
   +--> FSQ quantizer
   |
tokens = (num_tokens=2, token_dim=32)
   |
   +--> g1_kin decoder  -> 未来 G1 运动重建（训练辅助）
   +--> g1_dyn decoder  -> joint action（真正控制输出）
```

主干代码证据：

- `UniversalTokenModule` 定义与量化器初始化：`gear_sonic/trl/modules/universal_token_modules.py:33-240`
- encoder / decoder 动态实例化：`gear_sonic/trl/modules/universal_token_modules.py:242-399`
- encoder mask / 多 encoder 路由：`gear_sonic/trl/modules/universal_token_modules.py:496-540`
- encode / FSQ / decode 主流程：`gear_sonic/trl/modules/universal_token_modules.py:652-918`
- aux loss 输入组织：`gear_sonic/trl/modules/universal_token_modules.py:925-1078`

### 3.1 一个非常重要的阅读原则

只看 `all_mlp_v1.yaml` 会被误导。

原因是：

- `all_mlp_v1.yaml` 给的是一个 **通用拼装模板**
- `sonic_release.yaml` 会继续覆盖它的 `g1 / teleop / smpl` 输入输出定义

所以，**真实生效配置一定要以 experiment 配置合成后的结果为准**。对发布版来说，应以

`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`

为最终依据。

---

## 4. G1 路径：当输入格式为 `g1` 时，真实走的是什么

下面只分析 **发布版 `sonic_release`** 的真实路径。

### 4.1 G1 输入观测到底是什么

`sonic_release.yaml` 覆盖后的 `g1` encoder 输入是：

```yaml
inputs: ["command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"]
```

command_multi_future_nonflat 内部含有 joint_position 和 joint_velocity 
对应文件：

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml:84-107`
- `gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml:1-13`

#### 4.1.1 `command_multi_future_nonflat`

来源：

- 绑定：`gear_sonic/config/manager_env/observations/terms/command_multi_future_nonflat.yaml:1-6`
- observation 函数：`gear_sonic/envs/manager_env/mdp/observations.py:570-587`
- 真正返回内容：`gear_sonic/envs/manager_env/mdp/commands.py:897-903`

这里有一个很容易踩坑的点：

- `observations.py` 的 docstring 容易让人误以为这是 “body positions”
- 但真正实现里，`command_multi_future` 返回的是

  `torch.cat([joint_pos_multi_future, joint_vel_multi_future], dim=1)`

也就是说它本质上是：

- 每个 future frame：`[29 维 joint_pos, 29 维 joint_vel]`
- 单帧维度：`58`
- 发布版 `num_future_frames = 10`，所以形状是：`[10, 58]`

这一点在 `token_losses.py` 里也被明确写死：

- `gear_sonic/trl/losses/token_losses.py:71-75`

其中直接写了：

- `command_multi_future_nonflat: [..., num_future, 58] = [dof_pos(29), dof_vel(29)]`

#### 4.1.2 `motion_anchor_ori_b_mf_nonflat`

来源：

- 绑定：`gear_sonic/config/manager_env/observations/terms/motion_anchor_ori_b_mf_nonflat.yaml:1-6`
- observation 函数：`gear_sonic/envs/manager_env/mdp/observations.py:1022-1043`

语义：

- 参考 motion root 相对 robot root 的姿态差
- 6D rotation 表示
- 发布版 10 个 future frame，所以形状是：`[10, 6]`

#### 4.1.3 一个关键覆盖结论

`all_mlp_v1.yaml` 默认其实还把 `command_z_multi_future_nonflat` 放进了 `g1` 输入，`g1_kin` 输出里也默认带它：

- `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml:37-63`

但 `sonic_release.yaml` 把它覆盖掉了：

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml:87-107`

所以对发布版来说：

- **G1 encoder 不吃 `z`**
- **G1 kinematic decoder 也不重建 `z`**

另外要注意：

- `unitoken_all_noz.yaml` 虽然名字叫 `noz`，但 tokenizer 组里仍然注册了 `command_z_multi_future_nonflat` 和 `command_z`
- 也就是说，**“某个 observation 出现在 tokenizer group 里” 不等于 “它真的被当前 experiment 的 encoder/decoder 消费”**

这比只读基础模板更接近真实行为。

### 4.2 G1 Encoder 的真实网络结构

配置入口：

- `gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml:1-13`
- 真正实例化逻辑：`gear_sonic/trl/modules/universal_token_modules.py:260-308`
- MLP 工厂：`gear_sonic/trl/modules/base_module.py:143-189, 278-303, 532-560`

#### 4.2.1 网络结构

`g1_encoder` 是一个纯 MLP：

```text
Input(10 x 64 = 640)
 -> Linear(640, 2048) + SiLU
 -> Linear(2048, 1024) + SiLU
 -> Linear(1024, 512) + SiLU
 -> Linear(512, 512) + SiLU
 -> Linear(512, 64)
 -> reshape -> (2, 32)
```

其中：

- 单帧输入维度：`58 + 6 = 64`
- 时间维：`10`
- `BaseModule` 会先把 `[10, 64]` flatten 成 `640`
- 输出总维度：`64`
- 再按 `num_output_temporal_dims = max_num_tokens = 2` reshape 成 `[2, 32]`

#### 4.2.2 为什么是 `2 x 32`

因为在 `UniversalTokenModule` 里：

- `num_fsq_levels = 32`
- `max_num_tokens = 2`

所以：

- `token_dim = 32`
- `token_total_dim = 64`

对应代码：

- `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml:29-36`
- `gear_sonic/trl/modules/universal_token_modules.py:216-239`

### 4.3 G1 Kinematic Decoder 的真实网络结构

配置入口：

- `gear_sonic/config/actor_critic/decoders/g1_kin_mf_mlp.yaml:1-15`
- 发布版输出覆盖：`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml:105-107`
- decode 逻辑：`gear_sonic/trl/modules/universal_token_modules.py:692-739, 891-906`

#### 4.3.1 网络结构

`g1_kin` 也是纯 MLP：

```text
Input token: (2, 32)
 -> flatten -> 64
 -> Linear(64, 2048) + SiLU
 -> Linear(2048, 1024) + SiLU
 -> Linear(1024, 512) + SiLU
 -> Linear(512, 512) + SiLU
 -> Linear(512, 640)
 -> reshape -> (10, 64)
 -> split:
      command_multi_future_nonflat      : (10, 58)
      motion_anchor_ori_b_mf_nonflat    : (10, 6)
```

#### 4.3.2 它在训练里负责什么

`g1_kin` **不是实际控制器输出**，它主要用于两类训练约束：

1. `g1_recon`
   - 直接重建 G1 future motion
   - 实现：`gear_sonic/trl/losses/token_losses.py:528-557`

2. `reencoded_smpl_g1_latent`
   - 先用 `g1_kin` 从 token 重建 G1 motion，再把这份重建结果重新送回 `g1` encoder
   - 希望 re-encode 后 latent 贴近原本 G1 latent
   - 触发逻辑：`gear_sonic/trl/modules/universal_token_modules.py:971-1000`
   - loss 实现：`gear_sonic/trl/losses/token_losses.py:776-810`

所以可以把 `g1_kin` 理解成：

- **训练期的“token 是否真的保留了运动学信息”的探针**
- 而不是部署期动作输出头

### 4.4 G1 Dynamic Decoder 的真实网络结构

配置入口：

- `gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml:1-15`
- decoder 维度拼装逻辑：`gear_sonic/trl/modules/universal_token_modules.py:329-348`
- action 选择逻辑：`gear_sonic/trl/modules/universal_token_modules.py:908-923`

#### 4.4.1 输入是什么

`g1_dyn` 的输入是：

```yaml
inputs: ["token_flattened", "proprioception"]
outputs: ["action"]
```

也就是：

1. `token_flattened`
   - 来自 FSQ 之后的 token
   - `2 x 32 = 64`

2. `proprioception`
   - 发布版里就是 `actor_obs`
   - 来自 `local_dir_hist.yaml`

#### 4.4.2 `actor_obs` 的维度

发布版 policy obs 由下面几项构成：

- `gravity_dir`
- `base_ang_vel`
- `joint_pos`
- `joint_vel`
- `actions`

对应配置：

- `gear_sonic/config/manager_env/observations/policy/local_dir_hist.yaml:1-45`

发布版又把历史长度设为：

- `actor_prop_history_length = 10`
- `actor_actions_history_length = 10`

对应配置：

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml:30-34`

按 G1 29 DOF 计算：

- `gravity_dir`: `3 x 10 = 30`
- `base_ang_vel`: `3 x 10 = 30`
- `joint_pos`: `29 x 10 = 290`
- `joint_vel`: `29 x 10 = 290`
- `actions`: `29 x 10 = 290`

所以：

- `actor_obs = 30 + 30 + 290 + 290 + 290 = 930`

因此 `g1_dyn` 的总输入维度是：

- `64 + 930 = 994`

#### 4.4.3 网络结构

发布版 `g1_dyn` 默认是：

```text
Input(994)
 -> Linear(994, 2048) + SiLU
 -> Linear(2048, 2048) + SiLU
 -> Linear(2048, 1024) + SiLU
 -> Linear(1024, 1024) + SiLU
 -> Linear(1024, 512) + SiLU
 -> Linear(512, 512) + SiLU
 -> Linear(512, 29)
```

输出就是 `action_mean`。

对发布版来说，`UniversalTokenModule.forward()` 最后真正返回的是：

- `decoded_outputs["g1_dyn"]["action"]`

对应代码：

- `gear_sonic/trl/modules/universal_token_modules.py:891-923`

#### 4.4.4 谁会在部署时真正被调用

部署时：

- 正常模式：encoder -> FSQ -> `g1_dyn`
- token 旁路模式：外部直接给 `(B, 2, 32)` token，跳过 encoder，只走 `g1_dyn`

对应代码：

- `Actor.rollout_with_tokens()`：`gear_sonic/trl/modules/actor_critic_modules.py:439-485`
- `UniversalTokenModule.forward_with_external_tokens()`：`gear_sonic/trl/modules/universal_token_modules.py:1103-1163`

结论就是：

- **控制输出头永远是 `g1_dyn`**
- `g1_kin` 只在训练辅助里出现

---

## 5. FSQ：这个仓库里到底怎么用、怎么训练

## 5.1 仓库里 FSQ 的配置到底是什么

FSQ 配置文件非常短：

- `gear_sonic/config/actor_critic/quantizers/fsq.yaml:1`

内容只有：

```yaml
_target_: vector_quantize_pytorch.FSQ
```

真正的参数是在 `all_mlp_v1.yaml` 里给的：

- `num_fsq_levels: 32`
- `fsq_level_list: 32`
- `max_num_tokens: 2`

对应：

- `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml:29-36`

然后在 `UniversalTokenModule` 里被展开成：

- 如果 `fsq_level_list` 是整数，就复制成长度 `num_fsq_levels` 的列表
- 即：`[32] * 32`
- 再实例化：`FSQ(levels=[32, 32, ..., 32])`

对应代码：

- `gear_sonic/trl/modules/universal_token_modules.py:216-239`

所以对发布版来说，FSQ token 空间是：

- `num_tokens = 2`
- `token_dim = 32`
- `flattened token dim = 64`
- 每个 token 的 32 个 scalar 维度都各自量化到 32 个 level

### 5.1.1 这 32 个 level 的“具体数值”是不是整数

这里要把两个概念分开：

1. `levels = 32` 这个配置值本身是整数
2. 每个 scalar 被分到的 `level index` 也是整数
3. 但 FSQ 输出给 decoder 的 `quantized code value` 在默认实现下不是整数，而是归一化后的离散值

在本仓库里，能直接看到的是：

- `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml` 里把 `fsq_level_list: 32`
- `gear_sonic/trl/modules/universal_token_modules.py` 里把它展开成 `levels=[32] * 32`

也就是说，**每个 token 的每个 scalar 维度，都有 32 个离散档位**。

真正“这些档位对应哪些数值”，是在依赖库 `vector_quantize_pytorch.FSQ` 里定义的。官方实现里：

- `self._levels = tensor(levels, dtype = int32)`：说明 `levels` 本身就是整数计数
- `codes_non_centered = (...) % self._levels`：说明每维的档位编号是整数 index
- `half_width = self._levels // 2`
- `return (zhat - half_width) / half_width`：把整数档位映射回真正的 code value

官方源码可见：

- <https://raw.githubusercontent.com/lucidrains/vector-quantize-pytorch/master/vector_quantize_pytorch/finite_scalar_quantization.py>

因为这个仓库没有覆写 `preserve_symmetry` 等参数，所以按官方默认路径理解，若某一维 `levels = 32`，则：

- 整数档位编号是 `0, 1, 2, ..., 31`
- 实际量化后的 code value 是
  `[(k - 16) / 16 for k in 0..31]`
- 也就是：
  `[-1.0, -15/16, -14/16, ..., -1/16, 0, 1/16, ..., 15/16]`

所以结论是：

- **整数的是 level index**
- **真正的量化输出值不是整数，而是 32 个固定离散实数点**

## 5.2 FSQ 在 forward 里怎么被调用

核心调用点只有这里：

- `gear_sonic/trl/modules/universal_token_modules.py:684-690`

```python
quantized_codes, _ = self.quantizer(latent)
```

几个关键点：

1. 仓库只使用 `quantized_codes`
2. 量化器的第二个返回值被直接丢弃
3. decoder 一律吃量化后的 token，而不是原始 latent

在主 `forward()` 里又分成三种模式：

1. `post_quantization`（默认）
   - 先 encode
   - 再 FSQ
   - 最后如果有 residual，就加在量化后的 token 上

2. `pre_quantization`
   - 先 encode
   - residual 先加在 latent 上
   - 再 FSQ

3. `pre_quantization_replace`
   - 直接用 residual 替换 encoder latent
   - 再 FSQ

对应代码：

- `gear_sonic/trl/modules/universal_token_modules.py:822-880`

## 5.3 FSQ 在这个仓库里“靠什么被训练”

这是最关键的一点。

### 结论先说

**FSQ 在本仓库里不是单独训练，不是先离线 tokenizer pretrain，再冻结后训策略。**

它是和 encoder / decoder / policy 一起在 PPO 里端到端训练的。

证据链：

1. `Actor` 在训练时要求 backbone 返回 aux losses
   - `gear_sonic/trl/modules/actor_critic_modules.py:202-242`

2. `UniversalTokenModule.forward(compute_aux_loss=True)` 会同时给出：
   - `action_mean`
   - `decoded_outputs`
   - `encoded_tokens`
   - `encoded_latents`
   - `aux_losses`
   - `aux_loss_coef`

3. `TRLAuxLossPPOTrainer` 最终优化的是：

   `PPO loss + aux_loss_scale * Σ(coef_i * aux_loss_i)`

   - `gear_sonic/trl/trainer/ppo_trainer_aux_loss.py:6-196`

4. PPO 主损失本身在父类里算
   - `gear_sonic/trl/trainer/ppo_trainer.py:1435-1443`

这意味着：

- `g1_dyn` 通过 PPO policy loss 反向影响 token
- `g1_kin` 通过 reconstruction / cycle loss 反向影响 token
- 所有跨模态 latent alignment loss 也会通过 latent 反向影响 token 之前的 encoder 表征

### 一个更直接的说法

FSQ 在这个仓库里的训练信号来源只有两大类：

1. **控制目标**
   - `g1_dyn -> action_mean -> PPO loss`

2. **辅助目标**
   - `g1_recon`
   - `g1_smpl_latent`
   - `g1_teleop_latent`
   - `teleop_smpl_latent`
   - `reencoded_smpl_g1_latent`

## 5.4 这个仓库里“没有”的 FSQ 训练项

从仓库代码本身能确认：

1. 没有单独的 FSQ pretrain 脚本
2. 没有 repo 内自定义的 commitment loss
3. 没有 repo 内自定义的 codebook loss
4. 没有 repo 内自定义的 entropy regularization
5. 没有 repo 内自定义的 EMA codebook update 逻辑
6. 没有把 quantizer 的第二输出拿来算额外 loss

换句话说，本仓库对 FSQ 的使用方式是：

- **把它当作 encoder latent 和 decoder 之间的离散瓶颈**
- **训练信号完全来自下游控制损失和重建/对齐损失**

## 5.5 FSQ 是否可冻结

`UniversalTokenModule` 提供了：

- `freeze_quantizer`

对应代码：

- `gear_sonic/trl/modules/universal_token_modules.py:91-97, 425-428`

但发布版配置没有打开它，所以默认是：

- **FSQ 不冻结**

## 5.6 Auxiliary losses 的真实列表和权重

发布版 auxiliary loss 组合：

- `gear_sonic/config/aux_losses/universal_token/g1_recon_and_all_latent.yaml:1-16`

权重如下：

| loss | 权重 | 作用 |
| --- | --- | --- |
| `g1_recon` | `0.01` | 用 `g1_kin` 重建 G1 future motion |
| `g1_smpl_latent` | `1.0` | 让 G1 latent 与 SMPL latent 对齐 |
| `g1_teleop_latent` | `1.0` | 让 G1 latent 与 teleop latent 对齐 |
| `teleop_smpl_latent` | `1.0` | 让 teleop latent 成为 SMPL 的 bridge teacher |
| `reencoded_smpl_g1_latent` | `1.0` | cycle consistency：SMPL->G1 重建后再 encode 回去 |

实现位置：

- `g1_recon`：`gear_sonic/trl/losses/token_losses.py:528-557`
- `g1_smpl_latent`：`gear_sonic/trl/losses/token_losses.py:589-626`
- `teleop_smpl_latent`：`gear_sonic/trl/losses/token_losses.py:629-670`
- `g1_teleop_latent`：`gear_sonic/trl/losses/token_losses.py:678-719`
- `reencoded_smpl_g1_latent`：`gear_sonic/trl/losses/token_losses.py:776-810`

## 5.7 encoder 采样对 FSQ 训练意味着什么

encoder 不是每个 env 每个 episode 都固定用同一个。

在 `TrackingCommand` 里，`encoder_index` 会按 `encoder_sample_probs` 随机采样：

- `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml:14-20`
- `gear_sonic/envs/manager_env/mdp/commands.py:2898-2924`

发布版初始概率是：

- `g1: 1.0`
- `teleop: 1.0`
- `smpl: 1.0`

而且默认不是严格 one-hot 训练：

- 采到 `smpl` 时，代码会额外把 `g1` 也置 1
- 还可能按概率把 `teleop` 也置 1

对应代码：

- `gear_sonic/envs/manager_env/mdp/commands.py:2926-2959`

这会直接影响 FSQ 的训练方式：

1. token 瓶颈会被多个模态共同使用
2. 跨模态 latent alignment loss 才有 paired sample 可算
3. G1 这条 encoder 在默认模式下其实承担了 “共享教师空间” 的角色

## 5.8 发布版 PPO / 优化超参数

基础 PPO 配置：

- `gear_sonic/config/algo/ppo_im_phc.yaml:7-43`

发布版覆盖：

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml:27-83`

可以整理成：

| 项目 | 值 |
| --- | --- |
| `num_envs` | `4096` |
| `num_learning_iterations` | `100000` |
| `num_steps_per_env` | `24`（覆盖了基础配置的 32） |
| `num_learning_epochs` | `5` |
| `num_mini_batches` | `4` |
| `actor_learning_rate` | `2e-5` |
| `critic_learning_rate` | `1e-3` |
| `clip_param` | `0.2` |
| `entropy_coef` | `0.01` |
| `value_loss_coef` | `1.0` |
| `desired_kl` | `0.01` |
| `max_grad_norm` | `0.1`（发布版覆盖） |
| `init_noise_std` | `0.05` |
| `std clamp` | `[0.001, 0.5]` |
| `aux_loss_scale` | 默认 `1.0`（trainer 未被额外覆盖） |

---

## 6. 训练侧和部署侧如何对应

## 6.1 训练侧的 `g1` 输入，部署时被拆成了哪些观测

训练里 `g1_encoder` 用的是：

- `command_multi_future_nonflat`：`[10, 58]`
- `motion_anchor_ori_b_mf_nonflat`：`[10, 6]`

部署侧在 `observation_config.yaml` 和 C++ registry 里把它拆开了：

- `motion_joint_positions_10frame_step5`：`290`
- `motion_joint_velocities_10frame_step5`：`290`
- `motion_anchor_orientation_10frame_step5`：`60`

对应文件：

- 配置：`gear_sonic_deploy/policy/release/observation_config.yaml:25-80`
- 真实 registry：`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:1703-1791`

所以语义映射是：

| 训练侧 | 部署侧 |
| --- | --- |
| `command_multi_future_nonflat` | `motion_joint_positions_10frame_step5 + motion_joint_velocities_10frame_step5` |
| `motion_anchor_ori_b_mf_nonflat` | `motion_anchor_orientation_10frame_step5` |
| FSQ token `(2, 32)` | `token_state` / flattened 64D token |

### 6.1.1 为什么是 `step5`

发布版 motion command 配置里：

- `target_fps = 50`
- `dt_future_ref_frames = 0.1`
- `num_future_frames = 10`

对应：

- `gear_sonic/config/manager_env/commands/terms/motion.yaml:5-30`
- 发布版覆盖：`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml:44-56`

所以每个 future frame 间隔：

- `0.1s`
- 在 50Hz motion lib 里正好就是 **5 个采样步**

这就是部署侧命名为 `10frame_step5` 的原因。

## 6.2 部署侧 encoder 选择和训练侧不完全一样

训练侧：

- `encoder_index` 是 tokenizer observation 里的一列向量
- 默认训练还是 multi-hot 逻辑

部署侧：

- `observation_config.yaml` 里叫 `encoder_mode_4`
- C++ `GatherEncoderMode()` 实际上把当前 mode id 放在第一个槽位，后面补零

对应代码：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:1687-1709`

所以部署侧的 mode 表示更接近：

- “一个 mode id + padding”

而不是训练侧那种完整 multi-hot `encoder_index` 语义。

## 6.3 ONNX 导出路径

Python 侧已经把这套 universal-token 模块拆成三种导出方式：

1. encoder + decoder 成对导出
   - `gear_sonic/utils/inference_helpers.py:64-180`

2. encoders-only 导出
   - `gear_sonic/utils/inference_helpers.py:200-332`

3. decoder-only 导出
   - 同文件后半段

这说明仓库设计上本来就把：

- `encoder`
- `FSQ bottleneck`
- `decoder`

看成可以被拆开部署的几个部件。

---

## 7. 两个很重要的实现差异 / 代码真相

## 7.1 `all_mlp_v1.yaml` 不是最终真相

如果你只盯着：

- `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml`

会得到一个“g1/teleop 都还吃 z 观测”的印象。

但发布版 experiment 最后把它们覆盖成了：

- `g1`: 不吃 `command_z_multi_future_nonflat`
- `teleop`: 不吃 `command_z`
- `g1_kin`: 不重建 `z`

所以调模型时一定要读 experiment 配置，而不是只读基础 actor_critic preset。

## 7.2 发布版部署配置文件顶部的总维度注释是旧的

`gear_sonic_deploy/policy/release/observation_config.yaml` 顶部写着：

- `Total dimension: 436 (64+12+116+116+116+12)`

但它下面启用的是：

- `his_*_10frame_step1`

而 C++ registry 里这些 10-frame 观测的真实维度是：

- `30 + 290 + 290 + 290 + 30 + 64 = 994`

对应代码：

- 配置注释：`gear_sonic_deploy/policy/release/observation_config.yaml:1-23`
- registry 真实维度：`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:1776-1791`

所以这里更应信 C++ registry，而不是 YAML 顶部注释。

---

## 8. 最后的压缩结论

如果只保留最核心的认知，这套 motion tracker 可以简化成下面四句话：

1. **`g1_encoder`** 把 10 帧的 G1 参考 joint pos/vel 和 root orientation 差编码成 `2 x 32` 的 latent token。
2. **FSQ** 把这个 latent 变成离散 token 瓶颈；在本仓库里它不是单训的，而是和 PPO + aux loss 一起端到端训练。
3. **`g1_kin`** 只负责训练期重建 future G1 motion，帮助 token 保持运动学信息。
4. **`g1_dyn`** 才是实际控制输出头；它把 `64D token + 930D actor_obs` 映射成 29 维 joint action。

---

## 9. 关于 FSQ 底层实现的边界说明

本仓库没有把 `vector_quantize_pytorch` 的源码 vendored 进来，本地环境里也没有装上这个包，所以这份文档里关于 FSQ 的“底层数学细节”只写到 **仓库代码可直接验证** 的部分。

如果你还想继续往下追 `vector_quantize_pytorch.FSQ` 自身的实现，可以看它的官方资料：

- 项目主页：<https://github.com/lucidrains/vector-quantize-pytorch>
- FSQ 论文：<https://arxiv.org/abs/2309.15505>

从官方 README 和论文的公开说明来看，FSQ 属于：

- 参数更少的 scalar quantization
- 使用 straight-through gradient
- 不依赖传统 VQ 的 codebook / EMA / commitment loss

这和本仓库里“只拿 quantized token 去做下游任务、不额外加 codebook 类 loss”的使用方式是吻合的。
