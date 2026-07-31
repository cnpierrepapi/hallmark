"""Render the compliance record as a page a person can read and file.

Deliberately self contained and boring. This is the one thing here that might
end up attached to an email or printed and put in a folder, so it carries no
scripts, no external stylesheet and no colour that fails on a monochrome
printer. The same markup serves the page and the download: what someone files
is exactly what they were shown.
"""

from __future__ import annotations

from html import escape
from typing import Any

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f5f2; color: #16150f;
       font: 15px/1.55 ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
.page { max-width: 900px; margin: 0 auto; padding: 48px 28px 96px; }
h1 { font: 600 30px/1.2 Georgia, "Times New Roman", serif; margin: 0 0 6px; }
h2 { font: 600 13px/1.4 ui-monospace, monospace; letter-spacing: .14em;
     text-transform: uppercase; margin: 44px 0 14px; padding-bottom: 8px;
     border-bottom: 1px solid #16150f; }
p { margin: 0 0 12px; }
.lede { font: 400 17px/1.55 Georgia, serif; max-width: 62ch; }
.meta { display: grid; grid-template-columns: 180px 1fr; gap: 6px 18px; margin: 0; }
.meta dt { color: #5c584a; }
.meta dd { margin: 0; word-break: break-all; }
.statement { border-left: 3px solid #16150f; padding: 14px 0 14px 18px;
             background: #ecebe5; margin: 18px 0; max-width: 70ch; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid #d9d6cc;
         vertical-align: top; }
th { font-weight: 600; color: #5c584a; text-transform: uppercase;
     letter-spacing: .08em; font-size: 11px; }
.hash { word-break: break-all; color: #45412f; font-size: 12px; }
.mark { display: inline-block; border: 1px solid #16150f; border-radius: 2px;
        padding: 1px 6px; margin: 0 4px 4px 0; font-size: 11px; white-space: nowrap; }
.mark.absent { border-style: dashed; color: #8a8578; border-color: #b9b4a6; }
.scroll { overflow-x: auto; }
ul { margin: 0; padding-left: 20px; }
li { margin-bottom: 8px; max-width: 72ch; }
.cta { display: inline-block; margin: 6px 0 0; padding: 10px 18px;
       border: 1px solid #16150f; text-decoration: none; color: #16150f;
       background: #fff; font-weight: 600; }
.cta:hover { background: #16150f; color: #f6f5f2; }
.foot { margin-top: 46px; padding-top: 16px; border-top: 1px solid #d9d6cc;
        color: #5c584a; font-size: 12px; }
a { color: #16150f; }
@media print {
  body { background: #fff; }
  .page { padding: 0; max-width: none; }
  .noprint { display: none; }
  h2 { page-break-after: avoid; }
  tr { page-break-inside: avoid; }
}
"""


def _rows(assets: list[dict[str, Any]], base: str) -> str:
    out = []
    for asset in assets:
        marks = "".join(f'<span class="mark">{escape(m)}</span>' for m in asset["marks"])
        marks += "".join(
            f'<span class="mark absent">no {escape(m.lower())}</span>'
            for m in asset["missing"]
        )
        size = f"{asset['size_bytes'] / 1024:,.0f} KB" if asset["size_bytes"] else ""
        check = base + escape(asset["check"]) if asset.get("check") \
            else f"{base}/api/verify-stored/{escape(asset['slug'])}"
        label = escape(asset.get("check_label") or "check this file")
        # Whether these marks were read during this request or recorded when
        # the file was delivered. Both are measurements off the file, but they
        # were taken at different moments and the sheet should not blur that.
        when = (
            f'<br><span class="hash">{escape(asset["measured"])}</span>'
            if asset.get("measured")
            else ""
        )
        # The decision and the note behind it. Both are inside the canonical
        # hash, so a reader can check them the same way they check the pixels,
        # and the sheet says who worded the note rather than letting a machine
        # written sentence pass as something a person typed.
        decision = asset.get("decision") or ""
        review = f'<strong>{escape(decision)}</strong>' if decision else "&mdash;"
        if asset.get("reason"):
            review += f'<br>{escape(asset["reason"])}'
        if asset.get("reason_source"):
            review += f'<br><span class="hash">{escape(asset["reason_source"])}</span>'

        out.append(
            f"""<tr>
  <td><strong>{escape(asset['title'])}</strong><br>
      <span class="hash">{escape(asset['media_type'])} &middot; {size}</span></td>
  <td>{escape(asset['model'])}</td>
  <td>{review}</td>
  <td>{marks or '<span class="mark absent">none recorded</span>'}{when}</td>
  <td class="hash">{escape(asset['sha256'])}<br>
      <a href="{check}">{label}</a></td>
</tr>"""
        )
    return "\n".join(out)


def _rejects(rows: list[dict[str, Any]]) -> str:
    """The refused attempts, listed rather than counted."""
    out = []
    for row in rows:
        failed = ", ".join(row.get("failed_checks") or [])
        detail = f'<br><span class="hash">failed: {escape(failed)}</span>' if failed else ""
        size = f"{row['size_bytes'] / 1024:,.0f} KB" if row.get("size_bytes") else ""
        score = "" if row.get("score") is None else f"{row['score']}"
        out.append(
            f"""<tr>
  <td>{escape(str(row.get('modality', '')))}<br>
      <span class="hash">{escape(str(row.get('model', '')))}</span></td>
  <td>{escape(str(row.get('reason', '')))}{detail}</td>
  <td>{escape(score)}</td>
  <td class="hash">{escape(str(row.get('sha256', '')))[:32]}<br>
      <span class="hash">{size}</span></td>
</tr>"""
        )
    return "\n".join(out)


def render(record: dict[str, Any], base: str = "", *, standalone: bool = False) -> str:
    """Build the sheet. The download and the page are the same document."""
    approval = record.get("approval") or {}
    applied = record.get("marks_applied") or {}

    pipeline = "\n".join(
        f"<tr><td>{escape(role)}</td><td><strong>{escape(name)}</strong></td>"
        f"<td>{escape(why)}</td></tr>"
        for role, name, why in record.get("pipeline", [])
    )

    counts = "".join(
        f"<dt>{escape(label)}</dt><dd>{n} of {record['asset_count']} assets</dd>"
        for label, n in applied.items()
    )

    limits = "\n".join(f"<li>{escape(text)}</li>" for text in record.get("limits", []))

    ledger = "\n".join(
        f"<tr><td>{escape(str(r.get('modality','')))}</td>"
        f"<td>{escape(str(r.get('model','')))}</td>"
        f"<td>{r.get('attempts',0)}</td><td>{r.get('accepted',0)}</td>"
        f"<td>{r.get('acceptance_rate',0):.0%}</td></tr>"
        for r in record.get("ledger", [])
    )

    # A session sheet lives at a different URL from a campaign's, so the record
    # says where its own download is rather than the shape being assumed here.
    # Assuming it sent every visitor's download link to a page that 404s.
    href = record.get("download_url") or f"/compliance/{record['run_id']}/download"
    download = (
        ""
        if standalone
        else f'<p class="noprint"><a class="cta" href="{base}{escape(href)}">'
        "Download this sheet</a></p>"
    )

    session_scope = record.get("scope") == "session"

    # The refused attempts. A session sheet already lists them in the asset
    # table with their notes, so it gets the count and a pointer upward rather
    # than the same rows printed twice.
    refused = record.get("rejected") or []
    reject_count = record.get("reject_count", len(refused))
    refused_heading = f"Refused attempts ({reject_count})"
    if reject_count == 0:
        refused_lede = "Nothing was refused in this record."
        refused_table = ""
    elif refused:
        refused_lede = (
            "Generated, checked, and not approved. Kept in storage unsigned, because "
            "nobody passed them. Each note is inside the same hash as the approval, "
            "so it cannot be reworded after the fact without the record failing."
        )
        refused_table = f"""  <div class="scroll">
  <table>
    <thead><tr><th>Attempt</th><th>Why it was not used</th><th>Checks</th>
    <th>Content hash</th></tr></thead>
    <tbody>
{_rejects(refused)}
    </tbody>
  </table>
  </div>"""
    else:
        refused_lede = (
            "Listed in the asset table above, each with the note filed against it. "
            "They are kept in storage unsigned, because nobody approved them, and "
            "every note is inside the same hash as the approval."
        )
        refused_table = ""

    heading = "Everything this browser has generated" if session_scope else "Campaign"
    runs_row = (
        f"    <dt>Runs</dt><dd>{record.get('run_count', 0)}</dd>\n"
        if session_scope
        else ""
    )

    body = f"""
<div class="page">
  <h1>Marking record</h1>
  <p class="lede">{escape(record.get('product') or 'Campaign')}</p>

  <div class="statement">{escape(record['statement'])}</div>
  {download}

  <h2>{escape(heading)}</h2>
  <dl class="meta">
    <dt>{'Session' if session_scope else 'Run'}</dt><dd>{escape(record['run_id'])}</dd>
{runs_row}    <dt>Audience</dt><dd>{escape(record.get('audience') or 'Not recorded')}</dd>
    <dt>Assets</dt><dd>{record['asset_count']}</dd>
    <dt>Approved by</dt><dd>{escape(str(approval.get('approver') or 'Not recorded'))}</dd>
    <dt>Approved at</dt><dd>{escape(str(approval.get('approved_at') or 'Not recorded'))}</dd>
    <dt>Approval note</dt><dd>{escape(str(approval.get('note') or 'None'))}</dd>
    <dt>Record</dt><dd>{escape(record.get('manifest_uri') or 'Not recorded')}</dd>
  </dl>

  <h2>Marks applied</h2>
  <dl class="meta">{counts}</dl>

  <h2>Assets</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>Asset</th><th>Model</th><th>Review</th><th>Marks</th>
    <th>Content hash (SHA-256)</th></tr></thead>
    <tbody>
{_rows(record['assets'], base)}
    </tbody>
  </table>
  </div>

  <h2>{escape(refused_heading)}</h2>
  <p>{escape(refused_lede)}</p>
{refused_table}

  <h2>Pipeline</h2>
  <div class="scroll">
  <table><tbody>
{pipeline}
  </tbody></table>
  </div>

  <h2>Attempts behind these assets</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>Modality</th><th>Model</th><th>Attempts</th><th>Accepted</th><th>Rate</th></tr></thead>
    <tbody>
{ledger or '<tr><td colspan="5">No attempts recorded.</td></tr>'}
    </tbody>
  </table>
  </div>

  <h2>What this sheet does not say</h2>
  <ul>
{limits}
  </ul>

  <p class="foot">Each hash covers the picture, the sound and the visible credit,
  and leaves out the provenance boxes themselves, so attaching or removing a
  credential does not change it. Drop any of these files into the checker and it
  recomputes the same number in front of you. Nothing on this page has to be
  taken on trust.</p>
</div>
"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marking record {escape(record['run_id'][:8])}</title>
<style>{CSS}</style></head><body>{body}</body></html>"""
