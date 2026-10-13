# tools/ — Obsidian 笔记 → Hugo 站点

`sync.py` 把仓库里的 Obsidian 笔记转换成 `docs/` 下的 Hugo 站点内容。
**笔记永远是唯一的内容真相来源，转换器只读不写笔记。**

## 日常流程

```bash
python3 tools/sync.py && hugo serve --source docs   # 本地预览 http://localhost:1313/llm/
python3 tools/sync.py                              # 只生成，不预览
```

首次 clone 后需要先拉主题子模块：`git submodule update --init --recursive`。

本地没有 Hugo 的话：`brew install hugo`（Linux/CI 见 `.github/workflows/gh-pages.yaml` 里的 .deb 安装步骤）。

## CLI

| 命令 | 作用 |
| --- | --- |
| `python3 tools/sync.py` | 生成 `docs/content/`，并清理掉上一轮已不再发布的生成文件 |
| `python3 tools/sync.py --check` | 只校验、不写文件；并比对现有产物是否已过期。CI 用它做前置守卫 |
| `python3 tools/sync.py --strict` | 把警告也当成失败（例如图片缺失时想卡住部署） |
| `python3 tools/sync.py --clean` | 先清空 `docs/content/` 再生成 |
| `python3 tools/sync.py --serve` | 生成后直接起 `hugo server` |

**退出码**：`0` 正常（允许有警告）；`1` 清单或内容有错。图片缺失只算警告 —— 因为漏画一张图不应该让整站发不出去，页面里会出现一个显眼的占位块。

## `status.json` 字段说明

发布永远是**显式**的：`scanRoots` 下没有被 `posts` / `sections.index_source` / `pages` 引用、也不在 `skipped` 里的 `.md` 会打印警告并且**不会**被发布。

### 顶层

| 字段 | 说明 |
| --- | --- |
| `schemaVersion` | 固定为 `1` |
| `site` | 站点级信息。`title` / `baseURL` 必须与 `docs/hugo.yaml` 一致，`--check` 会断言，避免两处配置各说各话 |
| `scanRoots` | 扫描哪些目录来发现「未被收录的笔记」 |
| `assetRoot` | 图片目录，默认 `images` |
| `missingImagePolicy` | `placeholder`（默认，插占位块）/ `drop` / `error` |
| `sections` | 专题列表，可嵌套（靠 `parent`） |
| `posts` | 文章列表 |
| `pages` | 非文章的固定页（`home` / `search`） |
| `skipped` | 明确声明「存在但不发布」的文件，用来消掉未收录警告 |

### section

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | ✅ | 清单内部的唯一标识；`posts[].section` 指向它 |
| `slug` | ✅ | URL 段，小写 ASCII kebab-case |
| `parent` | | 父 section 的 `id`，用于嵌套（本项目里 `slime` 的 parent 是 `infra`） |
| `index_source` | | 用作专题落地页正文的笔记；不填则生成一段占位说明 |
| `title` / `description` | ✅ | 卡片标题与摘要 |
| `date` | | `YYYY-MM-DD`，**通常不用写**：日期默认取笔记的编辑时间，这里只是兜底 |
| `tags` | | 会生成 `/tags/<tag>/` 页 |
| `pin` | | 置顶配色名，取值必须是 `docs/assets/css/extended/pin-cards.css` 里定义过的 |
| `outline` | | `true` 时，这个专题页会在正文之后**按一级专题分组**列出卡片（`Above All` 用它当全站索引） |
| `toc` | | 是否显示目录 |

### post

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `source` | ✅ | 相对仓库根的笔记路径 |
| `section` | ✅ | 所属 section 的 `id` |
| `slug` | ✅ | URL 段，小写 ASCII kebab-case |
| `title` / `description` | ✅ | 卡片与页面元信息 |
| `date` | | `YYYY-MM-DD`，同上，通常不用写 |
| `tags` | | 会生成 `/tags/<tag>/` 页 |
| `weight` | | **通常不用填**，首页顺序由 `sync.py` 自动生成；写了则覆盖自动值，用于强制排序 |
| `stub` | | `true` 时，源文件为空则生成「🚧 撰写中」占位正文 |
| `demoteH1` | | 默认 `true`，把正文里的一级标题降为二级（避免与页面标题打架） |

### 首页顺序与卡片日期（都不用手维护）

首页卡片的**顺序**和卡片上显示的**日期**由 `sync.py` 一起算出来（`resolve_edit_times()` + `compute_weights()`），顺序分三段：

1. **置顶卡片** —— `Above All`（`pin: intro`），靠 `sections` 数组里排第一 + 最小 weight 固定在顶部；
2. **一级专题**（没写 `parent` 的 section）—— 按 `status.json` 里 `sections` 数组的书写顺序。想换大板块次序就调数组，现在是 `Model Architecture → Pre-Training → Mid-Training → Post-Training → LLM Infra`；
3. **二级专题（写了 `parent` 的）+ 文章** —— **混在同一条时间轴上**，按编辑时间新 → 旧。所以 `MidTraining Overview`（文章）会排在 `Slime 101`（二级专题）前面。

同一时间的：二级专题在前、文章在后，再同则按清单里的书写顺序（所以一个系列的多篇仍按清单顺序读下来）。

角标也跟着分两级：一级专题是淡蓝的 `Section`，二级专题是灰蓝的 `Sub-Section`（配色见 `docs/assets/css/extended/pin-cards.css`；二级的标记由 `sync.py` 依 `parent` 写进 frontmatter 的 `sub`）。

> ⚠️ 自动生成的 weight 从 **1** 开始编，不用 0：Hugo 把 `weight = 0` 当成「没设」，会把那张卡片扔到所有设了 weight 的页面**后面** —— 第一张卡片掉到最末尾。踩过一次。

**「编辑时间」怎么来的**（`source_edit_time()`）：

| 情况 | 取什么 |
| --- | --- |
| 工作区还有未提交改动（含刚新建、还没 `git add` 的文件） | 文件 mtime —— 本地刚写完就能立刻浮到前面 |
| 已提交、工作区干净 | git 里**内容最后一次改动**的提交时间（日期按**提交者所在时区**算，见下） |
| git 拿不到（浅克隆、不是 git 仓库） | 文件 mtime |
| 文件本身就不在 | 清单里的 `date` |

两条容易踩的坑，都在代码里处理掉了：

- **不能直接用 mtime 排序**。CI 每次都是全新检出，所有文件的 mtime 都是检出那一刻，按 mtime 排会退化成随机顺序。提交时间在克隆之后仍然稳定。
- **改名/挪目录不算「编辑」**。`_content_stamps()` 用一次带 `-M` 的 `git log --name-status` 把改名链串起来（新路径继承旧路径的历史），只有 R100（纯改名）会被跳过；改名同时改了内容仍然算一次编辑。所以把 `Advanced LLMs 5 (MoE).md` 挪成 `Advanced LLMs 4 (MoE).md` 不会让这张卡片假装今天刚更新过。

代价：**CI 必须检出全历史**（workflow 里 `fetch-depth: 0`）。浅克隆会被 `sync.py` 检出并打警告。

日期为什么按提交者时区算：CI 跑在 UTC，如果按**构建机器**的时区取「今天是几号」，你在 UTC+8 的凌晨提交的笔记会被记成前一天（真的踩过：00:00 提交、线上显示前一天）。git 提交里存着作者当时的时区偏移，用它算出来的日期本地和 CI 完全一致，跟构建环境在哪台机器、哪个时区无关。

写进 frontmatter 的日期**一定带偏移量**（`2026-10-14T00:00:33+08:00`），不写裸的 `2026-10-14`：裸日期会被 Hugo 按站点时区当成当天 00:00，对 UTC+8 的作者来说那是 8 小时后 —— 落在「未来」的页面会被 `buildFuture: false` **直接删掉**，表现是刚提交的几篇连着它的专题卡片一起从首页消失（也踩过）。`docs/hugo.yaml` 里的 `timeZone: Asia/Shanghai` 同理，管的是日期解析与显示用哪个时区。

`status.json` 里的 `date` 现在只是「文件不在 / git 拿不到时间」时的兜底，**可以不写**（`--check` 不再把它当必填）；卡片日期一律以笔记的编辑时间为准。

需要强制某篇排前面时，才在 `status.json` 里给它单独写 `weight`（写了就覆盖自动值）。

### 加一篇新笔记

1. 在 `model_arch/`、`infra/`、`mid-training/` 等目录里写好 `.md`。
2. 在 `status.json` 的 `posts` 里加一条：`source` 指路径、`slug` 定 URL、指定 `section`（没有对应专题就先在 `sections` 里加一个）。`date` 可以不写 —— 卡片日期和首页位置都用笔记的编辑时间。
3. `python3 tools/sync.py`。忘了加会被警告提示。

## 转换规则

笔记是 Obsidian 格式，转换器只改这些必要的地方，其余内容（包括公式、缩进、空行）原样透传：

| Obsidian | 转换结果 |
| --- | --- |
| `![[name.png]]` | `![name.png](name.png)` + 图片拷进该页的 page bundle；图片缺失时插入 `⚠️ 图片缺失` 占位块 |
| `==高亮==` | `<mark>高亮</mark>` |
| `[文字](./a.md)` | `[文字]({{< relref "…" >}})`，站点换路径也不会 404 |
| 正文里的 `# 一级标题` | 降为 `##` |

以下**不需要**转换，由 Goldmark / Hugo 自己处理：`[^1]` 脚注、裸 URL 自动成链、`$...$` 与 `$$...$$` 公式。

> ⚠️ **公式依赖 `docs/hugo.yaml` 里的 Goldmark passthrough 扩展，不要删。**
> 转换器只负责把公式**原样**写进 markdown，但 Goldmark 默认还会继续处理它，实测会踩两个坑：
> 一是公式里的 `_` 被当成斜体标记（`\mathbf{p}_t ... \sum_{m=0}` 的第 1、2 个下划线被配成一对 `<em>`，整条公式渲染失败）；
> 二是 `typographer` 扩展把 `l'` 的撇号换成弯引号 `’`。
> 开启 `passthrough`（并把 `typographer` 关掉）后，公式在 markdown → HTML 这一步被完整隔离。
> 这条链路上 268 个公式曾坏掉 49 个，修好后逐字保留 268 个。

> ℹ️ **公式是构建期渲染的**，由 `docs/layouts/_markup/render-passthrough.html` 调 `transform.ToMath` 完成（需要 Hugo ≥ 0.132）。
> 曾经走的是浏览器端 KaTeX auto-render，踩了两个坑，所以换掉：
> 一是相邻的两个 `$$` 块会被输出成紧挨着的一段文本（minify 连两行之间的换行也吃掉），`$$$$` 连在一起后 auto-render 配不上对，整块公式退化成一行源码；
> 二是公式里出现 `<` 时（例如 `o_{i,<t}`），passthrough 是原样输出的，浏览器会把 `<t>` 当成标签解析，文本节点被打断，同样渲染失败。
> 构建期渲染后每条公式都是独立的 `<span class="katex">`，两个毛病一起消失，也不再需要 CDN 上的 `katex.min.js` / `auto-render.min.js`（只剩一份 `katex.min.css`）。
> 代价是 HTML 变大（`rl-for-llms` 单页 45KB → 552KB，gzip 后 40KB），换来的是无 JS 也能看公式、公式进搜索索引。

> ⚠️ **`extend_head.html` 里的 `katex.min.css` 版本必须和 Hugo 的公式引擎配套，不能随手降。**
> Hugo 用的是 Go 版 KaTeX 移植，它输出的类名是 KaTeX **0.18 起**的新命名：`katex-sizing` / `katex-rule` / `katex-accent` / `katex-stretchy`；
> 0.17 及更早的 CSS 里这几条叫 `sizing` / `rule` / `accent` / `stretchy`。
> 配错的表现很隐蔽：公式看着"渲染了"，但**下标不缩小**（`.katex-sizing` 匹配不到，只剩 `vlist` 的位移），
> 于是 `J_{\mathrm{GRPO}} = J_{\mathrm{clip}} - \beta J_{\mathrm{KL}}` 显示成 `JGRPO = Jclip − βJKL`，一堆同字号的字挤在基线上。
> 换版本时核对一遍：Hugo 输出里的那 4 个 `katex-*` 类，在目标 CSS 里都要有规则。

两个关键实现细节：

- **先切代码块再做任何重写**。`slime-training.md` 的 python 代码块里有 `if self.role == "critic":` 这类 `==` 比较运算符，不先隔离代码块就会被高亮替换改坏。
- **图片走 leaf page bundle + 相对路径**。站点挂在 `/llm/` 子路径下，写死的 `/images/x.png` 会解析到域名根目录而 404；相对路径由 PaperMod 的 `render-image.html` 通过 `.PageInner.Resources` 解析，自动带上正确前缀。
- **围栏一律写明语言**（ASCII 图写 ```` ```text ````）。Hugo 0.146 会把**没有语言标记**的围栏当成 GoAT 渲染成 SVG 矢量图，等宽文本会变成一张按容器宽度拉伸的图，字号完全失控。转换器会在 sync 阶段对未标注语言的围栏报警告。`HUGO_VERSION` 见 `.github/workflows/gh-pages.yaml`，与本地 `brew install hugo` 保持一致。
