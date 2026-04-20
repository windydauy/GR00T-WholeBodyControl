# G1/SMPL Lite Motion Tracker 改造计划

**Summary**
- 新版本不再沿用 `sonic_release` 直接改，而是新增一套 `lite` 训练配置，主目标是“适合从零训练的轻量化 G1+SMPL 版本”。
- 结构上保留 `g1_encoder + g1_kin + g1_dyn + smpl_encoder`，完全移除 `teleop/vr encoder` 及其相关 tokenizer obs、aux loss、采样分支配置；`FSQ` 保持 `2 x 32` 不变。
- actor 输入改成三路：`token_flattened`、`actor_obs`、`future target geometry`。其中 `actor_obs` 只补机器人当前 root 全局位姿历史，`future target geometry` 新增未来 `N=10` 帧的参考 root 全局位姿和四末端相对 root 位姿。
- reward 新增四末端局部位置/朝向 tracking，并在新的 lite reward 组合里替换掉旧的 `tracking_vr_5point_local`，避免重复奖励同一几何量。

**Public Interfaces / Config Changes**
- 在 `TrackingCommandCfg` 增加 `endpoint_body` 和 `endpoint_body_offset` 两个配置字段，默认固定为：
  `["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"]`
  和沿用现有 wrists offset、feet 零 offset 的四组偏移。
- 新增 observation terms：
  `robot_anchor_pose_w`，当前机器人 root 的 `world xyz + 6D rot`。
  `motion_anchor_pose_w_mf`，未来 `N` 帧参考 root 的 `world xyz + 6D rot`，展平后输入 actor。
  `endpoint_pose_root_local_mf`，未来 `N` 帧四末端相对参考 root 的 `xyz + 6D rot`，展平后输入 actor。
- 新增 reward terms：
  `tracking_endpoint_local_pos`
  `tracking_endpoint_local_ori`
- 新增 lite 配置族：
  一个新的 `G1+SMPL lite` experiment config。
  一个新的 tokenizer obs config，只保留 `g1`/`smpl` encoder 所需项，并加入 actor 额外几何输入项。
  一个新的 policy obs config，在 `actor_obs` 中加入 `robot_anchor_pose_w` 历史。
  一个新的 aux loss config，只保留 `g1_recon`、`g1_smpl_latent`、`reencoded_smpl_g1_latent`。
  一个新的 reward composition config，用新的 endpoint rewards 替代旧的 `tracking_vr_5point_local`。

**Implementation Changes**
- 模态裁剪与配置分层：
  新增 lite experiment，而不是覆写现有 `sonic_release`。
  新 experiment 只激活 `g1` 和 `smpl` encoder；`encoder_sample_probs` 只包含这两路。
  `teleop` 相关 tokenizer terms、decoder inputs、aux losses、sampling 概率全部从 lite config 中删掉。
  保留当前 `smpl sampled => g1 also active` 的 latent alignment 逻辑；`teleop` 已有空值保护，不需要额外改采样代码。
- actor 观测重构：
  `actor_obs` 从现有 `930` 维扩成 `1020` 维。
  增量只来自 `robot_anchor_pose_w` 的 `10 x (3+6) = 90` 维历史；critic obs 语义不改。
  新增 `future target geometry` 两个 tokenizer terms：
  `motion_anchor_pose_w_mf = 10 x 9 = 90`
  `endpoint_pose_root_local_mf = 10 x 4 x 9 = 360`
  `g1_dyn.inputs` 改为：
  `["token_flattened", "proprioception", "motion_anchor_pose_w_mf", "endpoint_pose_root_local_mf"]`
  因为 decoder 额外输入是通过 `decode_input_dict.update(tokenizer_obs)` 注入，所以这两路必须做成 tokenizer obs，而不是 policy obs。
  新增 observation term 时统一采用无噪声默认值；v1 不给 root/global 和 future geometry 加 observation noise。
  v1 不改 `g1_encoder` 输入，避免同时改 token bottleneck 学习目标。
- reward 改造：
  在 command 侧缓存四末端当前/未来的 reference 与 robot `pos_w` / `quat_w`，并提供 root-local 变换所需属性。
  新位置 reward：把 reference 和 robot 四末端都变到各自 root local frame，比较 `xyz` 误差，用 Gaussian kernel。
  新朝向 reward：把 reference 和 robot 四末端都转到各自 root local frame，比较 quaternion angular error。
  新 lite reward 组合中移除 `tracking_vr_5point_local`，加入：
  `tracking_endpoint_local_pos`，默认 `weight=1.0, std=0.1`
  `tracking_endpoint_local_ori`，默认 `weight=1.0, std=0.4`
  其余 anchor/body/feet_acc/action regularization 奖励先保持不变。
- 轻量化网络：
  保持 `FSQ = 2 tokens x 32 dim` 不变。
  `g1_encoder`、`g1_kin`、`smpl_encoder` 的 hidden dims 统一从 `[2048, 1024, 512, 512]` 改成 `[1024, 512, 256, 256]`。
  `g1_dyn` 和 critic 的 hidden dims 从 `[2048, 2048, 1024, 1024, 512, 512]` 改成 `[1024, 1024, 512, 512, 256, 256]`。
  加上新 actor 几何输入后，`g1_dyn` 输入约变为 `64 + 1020 + 450 = 1534` 维；按这套宽度其参数量约 `3.61M`，明显低于“大网络 + 新输入”时的约 `11.29M`。
  v1 不减少层数，只减宽度，优先保训练稳定性和配置兼容性。

**Test Plan**
- 配置层：
  组合新的 lite experiment，确认只实例化 `g1`/`smpl` encoders，没有任何 `teleop` obs、loss、decoder 引用残留。
  检查 `g1_dyn` 最终输入维度与 observation dims 一致，`actor_obs=1020`，新增 future geometry=450。
- observation 层：
  为 `robot_anchor_pose_w`、`motion_anchor_pose_w_mf`、`endpoint_pose_root_local_mf` 写 shape 与数值基准测试。
  重点验证末端 local pose 使用的是 `endpoint_body/offset` 而不是 legacy `reward_point_body`。
- reward 层：
  构造零误差 case，验证两个新 reward 接近 `1`。
  分别注入末端位置扰动和朝向扰动，验证 reward 单调下降，且位置/朝向项互不串扰。
- 模型层：
  对新 lite actor/critic 做一次实例化和 forward smoke test，确认 `UniversalTokenModule` 能消费新的 tokenizer obs，`g1_dyn` 输出 action shape 正确。
- 训练层：
  做一次最小训练 smoke run，目标不是收敛，而是验证数据流、aux loss、reward、采样分支、checkpoint 保存全部跑通。
  记录初始化后的参数量摘要，确认与旧配置相比满足“显著减参”的预期。

**Assumptions / Defaults**
- 主实现目标是 `G1 + SMPL lite`，不是 `G1-only`；`G1-only` 作为后续 ablation，再通过移除 `smpl encoder + smpl aux losses` 单独做。
- 新增 actor 几何分支表示的是“参考未来目标”，不是机器人当前状态；机器人当前状态只在 `actor_obs` 中补 root 全局位姿历史。
- root 使用 `world xyz + 6D rotation`；四末端使用 `root-local xyz + 6D rotation`。
- 不修改现有 `sonic_release`、`tracking/base_5point_local_feet_acc` 和旧 tokenizer config 的行为；lite 路线通过新增配置并行存在，保证旧实验可复现。
