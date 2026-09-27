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
- **可选**：游戏上方的置顶小窗、中文语音播报。置顶小窗在 Windows 10 2004 及更新版本上不会被截图拍到（也不会出现在 OBS、Discord 的画面里）；更老的系统上它会放在左上角，别把它拖到棋盘、备战席或商店上。
- **赛后复盘**：每局自动记日志，`tft-advisor review --llm` 让 Claude 帮你复盘。
- **离线也能用的部分**：概率计算器、规则引擎、演示模式。

## 工作原理

```
截屏 (mss)  ->  Claude 视觉识别 (结构化 JSON)  ->  局面追踪 (合并/纠错/记录对手)
            ->  本地引擎 (经济、搜牌概率、装备、阵容匹配)  ->  规则建议 (立即显示)
            ->  Claude 策略 (更完整的建议)  ->  看板 / 小窗 / 语音
```

- 只截图、只读，**不会向游戏发送任何键盘鼠标操作，也不读游戏内存**。
- Claude 调用集中在 `tft_advisor/llm.py`：结构化输出、自适应思考、服务端拒答回退、系统提示缓存（赛季数据按缓存价计费，调用间隔超过 5 分钟时自动改用 1 小时缓存）、每分钟调用上限。

## 安装（Windows）

1. 从 python.org 安装 **64 位 Python 3.12**（推荐；3.11 到 3.14 都能用）。安装界面里勾选 "Add python.exe to PATH"（默认没勾）。
2. 下载代码：打开 https://github.com/YilouWang/YilouWang ，点 Code → Download ZIP，解压后进入里面的 **`tft-advisor` 文件夹**，在文件夹空白处右键「在终端中打开」（或按住 Shift 右键「在此处打开 PowerShell 窗口」）。然后：

   ```powershell
   py -3.12 -m venv .venv
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   .venv\Scripts\Activate.ps1
   python -m pip install -e ".[all]"
   ```

   - `py -3.12` 是 Python 自带的启动器，装的是别的版本就改成对应数字（比如 `py -3.13`）。
   - 第二行只需要执行一次，不需要管理员（问是否更改时输入 Y 回车）；不执行的话第三行会报「无法加载文件 ...Activate.ps1，因为在此系统上禁止运行脚本」。
   - 不想改执行策略也可以不激活，直接用虚拟环境里的程序：`.venv\Scripts\python.exe -m pip install -e ".[all]"`，以后用 `.venv\Scripts\tft-advisor.exe run` 启动（其他命令同理）。
   - `[all]` 包括可选的 OCR、语音（pyttsx3）和 pytest。OCR 在 Python 3.12 及以下用 rapidocr-onnxruntime，3.13 起用新版 rapidocr。只想要核心功能用 `python -m pip install -e .`。

3. 设置 Claude API Key（在 console.anthropic.com 创建）：

   ```powershell
   setx ANTHROPIC_API_KEY "sk-ant-..."
   $env:ANTHROPIC_API_KEY = "sk-ant-..."
   ```

   第一行永久保存（以后新开的窗口都有），第二行让当前窗口马上生效，不用重开。没有 Key 也能跑演示、概率计算器和规则引擎，但无法自动识别画面。

4. **以后每次新开 PowerShell**：先 `cd` 到 `tft-advisor` 文件夹，再运行 `.venv\Scripts\Activate.ps1`（或者不激活，直接运行 `.venv\Scripts\tft-advisor.exe`）。提示「无法将“tft-advisor”项识别为 cmdlet、函数...」就是忘了这一步。

5. 游戏设置：**窗口模式选「无边框」**（全屏独占模式下截图和置顶小窗可能不工作），推荐 16:9 分辨率。不需要以管理员身份运行本工具。

## 快速开始

```powershell
tft-advisor doctor          # 检查依赖、API Key、截图、数据（加 --api 会真实调用一次 Claude）
tft-advisor demo            # 用一局预先写好的 S18 对局演示整个流程（不需要游戏和 Key）
tft-advisor calibrate --delay 5   # 5 秒后截图并画出识别区域，检查是否对齐
tft-advisor run             # 正式开始，自动打开看板
```

反馈问题时可以把体检结果存成文件附上：`tft-advisor doctor > doctor.txt`（按控制台编码保存，中文不会乱码）。

### 第一次正式用之前

1. `tft-advisor data update`：提前下载最新赛季数据（会显示下载进度）。不提前下载也可以：`run` 会先用内置的 18.3 快照马上开始，同时在后台下载，下次启动生效；下载失败就继续用快照，6 小时后再自动重试。
2. `tft-advisor doctor --api`：确认 API Key、截图、数据都是 OK，并真实调用一次 Claude 识别和策略（约 0.1 美元），提前发现 Key 无效、模型不可用或请求被拒这类问题。
3. 进一局普通模式，`tft-advisor calibrate --delay 5` 后切回游戏，打开生成的 `%USERPROFILE%\.tft_advisor\calibrate.png`（在 PowerShell 里 `ii $HOME\.tft_advisor\calibrate.png` 直接打开）看框是否对齐。16:10 或 3:2 屏幕（很多笔记本）上框整体偏移、而游戏界面贴着屏幕边缘时，先执行 `$env:TFT_ADVISOR_HUD_LAYOUT = "anchored"` 再 calibrate 一次；对齐了就用 `setx TFT_ADVISOR_HUD_LAYOUT anchored` 永久设置。
4. `tft-advisor run`，在准备阶段按一次 F6，看看识别出的金币、等级、商店是否正确（识别错了可以在看板里手动修正）。
5. 打完第一局看一眼看板上的费用估算，再决定要不要换更便宜的模型或关掉自动读商店。

看板默认地址是 `http://127.0.0.1:8765`。想用手机看：

```powershell
tft-advisor run --host 0.0.0.0
```

终端会打印一行 `手机访问: http://192.168.x.x:8765/?token=...`，手机和电脑在同一个 Wi-Fi 下打开即可（`看板:` 那一行的 127.0.0.1 地址只能在这台电脑上用）。口令保存在 `%USERPROFILE%\.tft_advisor\dashboard_token`（改过 `[data] cache_dir` 时在那个文件夹里），每次启动都一样，手机上可以收藏这个链接；想换口令（旧链接失效）就用 `tft-advisor run --host 0.0.0.0 --new-token`，或者删掉这个文件。

**手机看板打不开**时按顺序检查：

1. 第一次用 `--host 0.0.0.0` 启动时，Windows 防火墙会弹窗询问是否允许 Python（显示的是 `...\Python312\python.exe`，不是虚拟环境里的那个）。点「允许访问」需要管理员密码；点了「取消」会生成一条阻止规则，以后也不会再弹窗，要到「控制面板 → Windows Defender 防火墙 → 允许应用通过防火墙」里把 Python 的阻止规则删掉或改成允许。
2. 新连的 Wi-Fi 默认是「公用网络」：在 Wi-Fi 属性里改成「专用网络」，或者在上面的防火墙设置里把 Python 的「公用」也勾上。
3. 关掉 VPN、加速器或代理软件的 TUN 模式（这时终端打印的可能是虚拟网卡的地址），也不要连路由器的访客网络（访客网络里设备互相访问不了）。
4. 没有管理员权限时，改用第二块屏幕打开电脑上的 127.0.0.1 看板。

重启程序时，已经打开的看板页面会自己重新连上，不会再多开一个标签页。

## 游戏中怎么用

| 热键 | 作用 |
|---|---|
| F6 | 立即完整分析一次 |
| F7 | 我正在看别人的棋盘：记录这个人的阵容 |
| F8 | 自动分析开关 |
| F9 | 读商店，告诉我买什么 |

- 笔记本上 F6 到 F9 默认可能是亮度、音量这类功能键，要按住 Fn（或打开 Fn 锁）。
- 程序运行时这几个键被全局占用，游戏和其他程序（比如浏览器里 F6 跳到地址栏）收不到它们。想换键在 `config.toml` 的 `[hotkeys]` 里改，例如 `analyze = "ctrl+shift+f6"`；启动时提示「热键 F6 注册失败（已被其他程序占用）」也是这样换一个。

- 自动模式下，回合切换后约 1 秒自动分析；刷新商店时自动读商店。自动分析只看**游戏窗口，并且游戏在前台**的时候：找不到游戏窗口、窗口最小化或者你切到了别的程序（浏览器、聊天软件），自动分析会暂停，不会把桌面或别的程序的画面发给 Claude。
- 双屏时，你点到另一块屏幕上的看板会让游戏失去前台，自动分析会暂停到你点回游戏为止；不想这样可以设 `[capture] require_foreground = false`（这时如果游戏窗口被别的窗口挡住，挡住的内容也会被截进去）。
- 搜牌时商店刷新很快，自动读商店最多每 3 秒一次（`shop_min_interval_s`），每分钟剩余调用次数不多时会先停掉自动读商店，把额度留给回合分析。
- 按 F6/F7/F9（或点看板按钮）的那一刻就会截图，所以按完 F7 马上切回自己的棋盘也没关系。上一次识别还没完成时，记录对手会排队（连按几个对手也不会丢），读商店会在当前分析之后补读。这几个热键在找不到游戏窗口时会截整块屏幕。
- 先显示规则引擎的建议，Claude 的完整建议想好后再替换上去；等 Claude 的时候，读商店和记录对手不用排队。同一回合里读商店、记录对手不会把 Claude 的建议换掉，只把其中「买牌」一条按新商店更新（标记为「规则 + Claude」）。
- 看板上的按钮和热键功能相同，热键不可用时（比如被别的软件占用）直接点看板或手机。
- 看板上「需要你帮忙」一栏是助手请你做的事，做完点「我已切到他的棋盘，记录」。只有一条请求时，按 F7 也会按那个人的名字记录。
- 识别错了（比如金币读错），在看板的「手动修正」里改，之后的建议会用你改的值。
- 「提问」框可以问任何问题，Claude 会结合当前局面回答。

## 配置

`tft-advisor init` 会生成 `%USERPROFILE%\.tft_advisor\config.toml`（带中文注释，`notepad $HOME\.tft_advisor\config.toml` 打开编辑），程序会自动读这个文件。放在别的位置时用 `tft-advisor --config 路径 run`（`--config` 也可以写在命令后面），或者设置环境变量 `TFT_ADVISOR_CONFIG`。为了安全，**当前目录下的配置文件不会被自动读取**；`run` 和 `demo` 启动时会打印用的是哪个配置文件。

配置文件要存成 UTF-8（记事本「另存为」里选 UTF-8）。**路径用单引号或正斜杠**：`comps_file = 'C:\Users\你\我的阵容.json'` 或 `"C:/Users/你/我的阵容.json"`；双引号里的 `\` 是转义字符（`"D:\tft\new.json"` 里的 `\t`、`\n` 会被读成制表符和换行）。相对路径按配置文件所在的文件夹算。写错的项（拼写、类型、范围、按键名）会在启动时用一行中文指出来；需要完整报错信息时设置环境变量 `TFT_ADVISOR_DEBUG=1`。常用项：

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

每次画面识别约 4 千输入 token（整张截图加几张局部放大图，1080p 下）+ 约 1 千输出 token，读商店只发底部一条，便宜得多；每次策略约 3 千输入 + 1 到 2 千输出；赛季数据作为系统提示会被缓存。按 Opus 5 价格，自动模式一局大约 3 到 6 美元（刷新商店很频繁时更多）；识别模型换成 Sonnet 5 大约能省一半以上。第一局建议盯一下看板上的费用估算。省钱办法：

- `auto = false`，只在关键回合按 F6；
- `shop_watch = false`，只在需要时按 F9；
- `max_calls_per_minute` 调低。

看板顶部会显示这次运行以来的 Claude 调用次数和按官方价格估算的费用（重启程序才清零，只是估算，以控制台账单为准）。

## 数据与更新

- **赛季数据**：从 CommunityDragon 下载 `zh_cn` 和 `en_us` 数据（缓存 24 小时，`tft-advisor data update` 强制更新）。`run` 启动时不等下载：先用缓存或内置的 S18 18.3 快照，过期的数据在后台更新，下次启动生效。下载失败时使用内置快照，6 小时内不再自动重试。
- **机制数值**（商店概率、卡池、经验表、连胜金币）：`tft_advisor/data/bundled/mechanics.toml`，当前是 18.3b 的数值。版本更新后如果变了，写一个只包含改动项的 TOML，在配置里用 `[data] mechanics_file = "..."` 指过去。
- **阵容库**：内置 `comps_set18.json`（TFTAcademy 18.3，45 套）。自定义阵容：

  ```json
  {"comps": [
    {"name": "我的阿狸", "style": "fast8", "tier": "A",
     "units": ["Ahri", "Morgana", "Sett", "Karma"],
     "carry": "Ahri", "carry_items": ["Jeweled Gauntlet", "Spear of Shojin"]}
  ]}
  ```

  把文件放在 `config.toml` 旁边，在配置里写 `[data] comps_file = '我的阵容.json'`（相对路径按配置文件所在的文件夹算）。名字中英文都可以。文件找不到、格式错误或者一套阵容都读不出来时，启动会提示原因并改用内置阵容库。

- **海克斯强化**：内置 `augments_set18.json`（249 个 S18 强化的中英文名、效果、品级、类别，以及 2026-09-26 MetaTFT 高分段 S 级名单的静态快照）。选强化时规则引擎会给出推荐和效果说明，Claude 也会拿到这几个选项的效果文字。重新生成：`python tools/build_augments.py --ddragon <Data Dragon 数据目录> --metatft <MetaTFT lookup.json>`。
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

隐私：

- 截图（包含同局其他玩家的游戏名）会发给 Anthropic 的 API 做识别。自动模式只发游戏窗口在前台时的画面；手动按热键时发的是按键那一刻的画面（找不到游戏窗口时是整块屏幕）。
- 对局日志只保存在你自己电脑的 `%USERPROFILE%\.tft_advisor\logs\`，每局一个文件，里面有同局玩家的名字和阵容。默认只保留最近 30 局（`[data] keep_game_logs`，0 = 全部保留），更早的会自动删除。演示模式的日志单独放在 `logs\demo\`，不会被 `review` 当成你的对局。
- `save_screenshots = true` 时保存的截图不会自动删除，调试完记得清理 `screenshot_dir`。

## 已知限制

- S18 起 TFT 迁移到了虚幻引擎，界面布局和旧版本不同。识别主要靠 Claude 看整张截图，对界面变化比较鲁棒；截图裁剪区域是按经验估的，用 `tft-advisor calibrate` 检查，偏得厉害时可以反馈。
- Riot 计划 2026-10-09 推出独立的 TFT 客户端，窗口标题可能会变：如果 `doctor` 找不到游戏窗口，在配置里改 `[capture] window_title`。找不到窗口时热键会截整块屏幕，但自动分析会暂停；想让自动分析也用整块屏幕，设 `[capture] use_window = false`（这时屏幕上的任何内容都可能被发去识别）。
- 视觉识别偶尔会读错星级或装备；关键数字（金币、等级、回合）可以装 OCR（在 `tft-advisor` 文件夹运行 `python -m pip install -e ".[ocr]"`，`[all]` 已经包含），并在配置里开 `ocr_crosscheck = true` 交叉校验（每次分析会多花一点时间）。装了 OCR 后，读商店（F9）会优先用本地 OCR，又快又不花钱。OCR 依赖的 onnxruntime 需要微软 VC++ 运行库：`tft-advisor doctor` 显示「缺少 VC++ 运行库」时安装 https://aka.ms/vs/17/release/vc_redist.x64.exe ，重新安装 OCR 没有用。
- 奖励机制很多的赛季（S18 的精灵 Wisps、各种召唤物、Lux 变体）识别难度更高，建议以看板为参考而不是绝对指令。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

架构和模块接口见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。所有测试离线运行，Claude 调用用 `tests/fakeapi.py` 的本地假服务器模拟。
