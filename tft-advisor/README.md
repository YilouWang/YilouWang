# TFT 实时助手 (tft-advisor)

个人自用的云顶之弈实时建议工具。它在你打游戏时截屏，用 Claude 视觉模型读出当前局面（回合、金币、等级、经验、血量、商店、棋盘、备战席、装备、其他玩家血量），再用本地的经济/概率引擎和 Claude 策略模型给出建议：这回合该存钱、升级还是搜牌，搜多少，买哪张卡，装备怎么合，玩什么阵容，谁在和你抢卡。

需要别人的信息时，它会请你配合：比如「请点开右侧玩家列表里 XXX 的棋盘，然后按 F7 记录」。你切过去按一下，它就把那个人的阵容记下来，算搜牌概率时会扣掉被别人拿走的卡。

> 当前内置数据：**S18「Enchanted Wilds」补丁 18.3**（英雄、羁绊、合成表、中英文名、TFTAcademy 18.3 阵容库）。联网时会自动从 CommunityDragon 更新到最新版本。

## 功能

- **自动识别局面**：回合切换时自动截图分析（也可以按 F6 手动分析）。
- **经济与节奏**：利息、连胜连败金币、标准升级节奏、关键搜牌点（3-2 / 4-1 / 4-2）、血量危险时 all-in。
- **搜牌概率**：按当前等级的商店概率和卡池剩余（扣掉你和被侦察到的对手持有的卡）计算：单格概率、一次刷新出现概率、花 20/40/60 金币升星的概率、期望花费；还会比较「直接搜」和「先升一级再搜」。
- **读商店**：刷新商店时自动读一遍（或按 F9），告诉你哪张该买。
- **装备**：散件能合什么、给谁、什么时候该直接合（slam）。
- **阵容**：根据你手里的卡、装备和对手的阵容推荐方向（内置 45 套 18.3 阵容，也支持自定义）。
- **侦察配合**：提醒你去看谁的棋盘，记录后自动计算被抢的卡。
- **看板**：浏览器页面，第二块屏幕或手机都能看；有按钮可以代替热键；可以直接打字问问题（比如「现在该不该转法师？」）。
- **可选**：游戏上方的置顶小窗、中文语音播报。
- **赛后复盘**：每局自动记日志，`tft-advisor review --llm` 让 Claude 帮你复盘。
- **离线也能用的部分**：概率计算器、规则引擎、演示模式。

## 工作原理

```
截屏 (mss)  ->  Claude 视觉识别 (结构化 JSON)  ->  局面追踪 (合并/纠错/记录对手)
            ->  本地引擎 (经济、搜牌概率、装备、阵容匹配)  ->  规则建议 (立即显示)
            ->  Claude 策略 (更完整的建议)  ->  看板 / 小窗 / 语音
```

- 只截图、只读，**不会向游戏发送任何键盘鼠标操作，也不读游戏内存**。
- Claude 调用集中在 `tft_advisor/llm.py`：结构化输出、自适应思考、服务端拒答回退、系统提示缓存（赛季数据只在第一次计费，后面按缓存价）、每分钟调用上限。

## 安装（Windows）

1. 安装 Python 3.11 或更新版本（安装时勾选 "Add Python to PATH"）。
2. 下载这个仓库，在 `tft-advisor` 目录打开 PowerShell：

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[all]"
   ```

   `[all]` 包括可选的 OCR（rapidocr）、语音（pyttsx3）和 pytest。只想要核心功能用 `pip install -e .`。

3. 设置 Claude API Key（在 console.anthropic.com 创建）：

   ```powershell
   setx ANTHROPIC_API_KEY "sk-ant-..."
   ```

   设置后重新打开 PowerShell。没有 Key 也能跑演示、概率计算器和规则引擎，但无法自动识别画面。

4. 游戏设置：**窗口模式选「无边框」**（全屏独占模式下截图和置顶小窗可能不工作），推荐 16:9 分辨率。

## 快速开始

```powershell
tft-advisor doctor          # 检查依赖、API Key、截图、数据
tft-advisor demo            # 用内置样例对局演示整个流程（不需要游戏和 Key）
tft-advisor calibrate --delay 5   # 5 秒后截图并画出识别区域，检查是否对齐
tft-advisor run             # 正式开始，自动打开看板
```

看板默认地址是 `http://127.0.0.1:8765`。想用手机看：

```powershell
tft-advisor run --host 0.0.0.0
```

终端会打印带访问口令的地址（`?token=...`），手机和电脑在同一个 Wi-Fi 下打开即可。

## 游戏中怎么用

| 热键 | 作用 |
|---|---|
| F6 | 立即完整分析一次 |
| F7 | 我正在看别人的棋盘：记录这个人的阵容 |
| F8 | 自动分析开关 |
| F9 | 读商店，告诉我买什么 |

- 自动模式下，回合切换后约 1 秒自动分析；刷新商店时自动读商店。
- 看板上的按钮和热键功能相同，热键不可用时（比如被别的软件占用）直接点看板或手机。
- 看板上「人工请求」一栏是助手请你做的事，做完点「我已切到他的棋盘，记录」。
- 识别错了（比如金币读错），在看板的「手动修正」里改，之后的建议会用你改的值。
- 「提问」框可以问任何问题，Claude 会结合当前局面回答。

## 配置

`tft-advisor init` 会生成 `~/.tft_advisor/config.toml`（带中文注释）。常用项：

```toml
[anthropic]
vision_model = "claude-opus-5"     # 想更快更省钱可改 "claude-sonnet-5"
strategy_model = "claude-opus-5"
vision_effort = "low"
strategy_effort = "medium"
max_calls_per_minute = 12

[advisor]
auto = true
comp_hint = ""                     # 想玩的阵容，比如 "Ashe" 或 "法师"

[ui]
overlay = false                    # 置顶小窗
voice = false                      # 语音播报
```

### 费用估算

每次画面识别约 3 千输入 token（图片）+ 约 1 千输出 token，每次策略约 3 千输入 + 1 到 2 千输出；赛季数据作为系统提示会被缓存。按 Opus 5 价格，自动模式一局大约 2 到 4 美元；识别模型换成 Sonnet 5 大约能省一半以上。省钱办法：

- `auto = false`，只在关键回合按 F6；
- `shop_watch = false`，只在需要时按 F9；
- `max_calls_per_minute` 调低。

看板顶部会显示本局 Claude 调用次数。

## 数据与更新

- **赛季数据**：启动时从 CommunityDragon 下载 `zh_cn` 和 `en_us` 数据（缓存 24 小时，`tft-advisor data update` 强制更新）。下载失败时使用内置的 S18 18.3 快照。
- **机制数值**（商店概率、卡池、经验表、连胜金币）：`tft_advisor/data/bundled/mechanics.toml`，当前是 18.3b 的数值。版本更新后如果变了，写一个只包含改动项的 TOML，在配置里用 `[data] mechanics_file = "..."` 指过去。
- **阵容库**：内置 `comps_set18.json`（TFTAcademy 18.3，45 套）。自定义阵容：

  ```json
  {"comps": [
    {"name": "我的阿狸", "style": "fast8", "tier": "A",
     "units": ["Ahri", "Morgana", "Sett", "Karma"],
     "carry": "Ahri", "carry_items": ["Jeweled Gauntlet", "Spear of Shojin"]}
  ]}
  ```

  在配置里 `[data] comps_file = "我的阵容.json"`。名字中英文都可以。

- 重新生成内置快照：`python tools/build_snapshot.py --en en_us.json --zh zh_cn.json`。

## 其他命令

```powershell
tft-advisor odds --level 8 --cost 4 --have 1 --star 2 --taken 2 --gold 60 --xp 20
tft-advisor review --llm            # 复盘最近一局
tft-advisor replay 截图目录           # 对保存的截图跑一遍分析（配合 save_screenshots = true）
tft-advisor data show --full        # 看当前赛季数据
```

## 合规与风险（请先读）

这个工具只截屏、只读 Riot 本地公开的 Live Client Data 接口（S18 虚幻引擎客户端上大概率已经没有这个接口，工具会自动跳过），**不会向游戏发送任何键盘鼠标操作，不注入、不读内存，也不预测下一个对手是谁**。但你需要知道 Riot 的 TFT 第三方工具政策（开发者门户 2026 年 8 月的版本）明确写了这些「不允许的用例」：

- "Scouting - tracking the champions opponents have on their boards."（侦察：记录对手棋盘上的英雄）
- "Apps that provide dynamic, real-time information." / "Apps that dictate player decisions."（提供动态实时信息、替玩家做决定的应用）
- "An app cannot make suggestions based on the player's current game state"（不能根据当前对局状态给建议）

这些条款针对的是向 Riot 注册的第三方产品。这个工具是你自己电脑上的私人程序，被动截屏目前没有已知的检测方式，Vanguard 也没有公开说明会拦截截屏。但它在对局中给出实时建议、记录对手阵容，**本质上和政策意图相违背，在排位赛中使用有被判定违规的风险，后果由你自己承担**。建议：

- 优先在普通模式、练习和自定义房间里用，把它当「陪练教练」，学会思路后自己打；
- 用 `tft-advisor review --llm` 做赛后复盘，这部分完全符合政策（只看自己的对局）；
- 不想在对局中看对手信息时，把配置里的 `scout_prompts = false`。

隐私：截图（包含同局其他玩家的游戏名）会发给 Anthropic 的 API 做识别；对局日志只保存在你自己电脑的 `~/.tft_advisor/logs/`。

## 已知限制

- S18 起 TFT 迁移到了虚幻引擎，界面布局和旧版本不同。识别主要靠 Claude 看整张截图，对界面变化比较鲁棒；截图裁剪区域是按经验估的，用 `tft-advisor calibrate` 检查，偏得厉害时可以反馈。
- Riot 计划 2026-10-09 推出独立的 TFT 客户端，窗口标题可能会变：如果 `doctor` 找不到游戏窗口，在配置里改 `[capture] window_title`，或者直接用整块屏幕截图（找不到窗口时会自动退回整屏）。
- 视觉识别偶尔会读错星级或装备；关键数字（金币、等级、回合）可以装 OCR（`pip install rapidocr-onnxruntime`），并在配置里开 `ocr_crosscheck = true` 交叉校验（每次分析会多花一点时间）。装了 OCR 后，读商店（F9）会优先用本地 OCR，又快又不花钱。
- 奖励机制很多的赛季（S18 的精灵 Wisps、各种召唤物、Lux 变体）识别难度更高，建议以看板为参考而不是绝对指令。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

架构和模块接口见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。所有测试离线运行，Claude 调用用 `tests/fakeapi.py` 的本地假服务器模拟。
