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
| `date` | ✅ | `YYYY-MM-DD` |
| `tags` / `weight` | | `weight` 为负会让专题卡片置顶，排在文章前面 |
| `pin` | | 置顶配色名，取值必须是 `docs/assets/css/extended/pin-cards.css` 里定义过的 |
| `toc` | | 是否显示目录 |

### post

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `source` | ✅ | 相对仓库根的笔记路径 |
| `section` | ✅ | 所属 section 的 `id` |
| `slug` | ✅ | URL 段，小写 ASCII kebab-case |
| `title` / `description` / `date` | ✅ | 卡片与页面元信息 |
| `tags` | | 会生成 `/tags/<tag>/` 页 |
| `weight` | | 同一列表内的排序，越小越靠前 |
| `stub` | | `true` 时，源文件为空则生成「🚧 撰写中」占位正文 |
| `demoteH1` | | 默认 `true`，把正文里的一级标题降为二级（避免与页面标题打架） |

### 加一篇新笔记

1. 在 `model_arch/`、`infra/` 等目录里写好 `.md`。
2. 在 `status.json` 的 `posts` 里加一条：`source` 指路径、`slug` 定 URL、`date` 必须显式给（这些笔记大多没有 git 历史，无法从别处推断）。
3. `python3 tools/sync.py`。忘了加会被警告提示。

## 转换规则

笔记是 Obsidian 格式，转换器只改这些必要的地方，其余内容（包括公式、缩进、空行）原样透传：

| Obsidian | 转换结果 |
| --- | --- |
| `![[name.png]]` | `![name.png](name.png)` + 图片拷进该页的 page bundle；图片缺失时插入 `⚠️ 图片缺失` 占位块 |
| `==高亮==` | `<mark>高亮</mark>` |
| `[文字](./a.md)` | `[文字]({{< relref "…" >}})`，站点换路径也不会 404 |
| 正文里的 `# 一级标题` | 降为 `##` |

以下**不需要**转换，由 Goldmark / KaTeX 自己处理：`[^1]` 脚注、裸 URL 自动成链、`$...$` 与 `$$...$$` 公式。

> ⚠️ **公式依赖 `docs/hugo.yaml` 里的 Goldmark passthrough 扩展，不要删。**
> 转换器只负责把公式**原样**写进 markdown，但 Goldmark 默认还会继续处理它，实测会踩两个坑：
> 一是公式里的 `_` 被当成斜体标记（`\mathbf{p}_t ... \sum_{m=0}` 的第 1、2 个下划线被配成一对 `<em>`，整条公式渲染失败）；
> 二是 `typographer` 扩展把 `l'` 的撇号换成弯引号 `’`。
> 开启 `passthrough`（并把 `typographer` 关掉）后，公式在 markdown → HTML 这一步被完整隔离。
> 这条链路上 268 个公式曾坏掉 49 个，修好后逐字保留 268 个。

两个关键实现细节：

- **先切代码块再做任何重写**。`slime-training.md` 的 python 代码块里有 `if self.role == "critic":` 这类 `==` 比较运算符，不先隔离代码块就会被高亮替换改坏。
- **图片走 leaf page bundle + 相对路径**。站点挂在 `/llm/` 子路径下，写死的 `/images/x.png` 会解析到域名根目录而 404；相对路径由 PaperMod 的 `render-image.html` 通过 `.PageInner.Resources` 解析，自动带上正确前缀。
- **围栏一律写明语言**（ASCII 图写 ```` ```text ````）。Hugo 0.146 会把**没有语言标记**的围栏当成 GoAT 渲染成 SVG 矢量图，等宽文本会变成一张按容器宽度拉伸的图，字号完全失控。转换器会在 sync 阶段对未标注语言的围栏报警告。`HUGO_VERSION` 见 `.github/workflows/gh-pages.yaml`，与本地 `brew install hugo` 保持一致。
