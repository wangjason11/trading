"""Landing review (sweep integration) mutants for PLAN_R step 2 — each applied alone, full suite."""
OV = "engine_v2/multitf/proximity_overlay.py"
OR = "engine_v2/pipeline/orchestrator.py"
MUT = [
    ("R-A the view's cap not localized (entity-absolute cap into a slice-local run)", OR,
     "cap=(None if cap_abs is None else int(cap_abs) - sb),",
     "cap=(None if cap_abs is None else int(cap_abs)),"),
    ("R-B the view drops the sub's cross-fib limit", OR,
     "log_prefix=f\"PROX_VIEW{sub.pool_idx}\", cross_fib_limits={0: sub.cross_fib_limit.localized(sb)},",
     "log_prefix=f\"PROX_VIEW{sub.pool_idx}\", cross_fib_limits=None,"),
    ("R-C the view's captured stdout re-prints nothing (every grep line hidden)", OV,
     '_VIEW_LOG_KEEP = re.compile(r"error|warning|skipping|unavailable|degenerate|pending|no sid", re.IGNORECASE)',
     '_VIEW_LOG_KEEP = re.compile(r"(?!)")'),
    ("R-D the record-start gate of a source's rungs reads the LATEST record start (max)", OV,
     "        return min(starts) if starts else None",
     "        return max(starts) if starts else None"),
    ("R-E the refresh before close(h - 1): a zone usable from h is pass-tested on h - 1's close", OV,
     "        if h > scan.scan_start:\n"
     "            passed = scan.close(h - 1)\n"
     "            if passed:\n"
     "                self.pass_log.append((key, h - 1, [r.key for r in passed]))\n"
     "        self._refresh(key, h, known)\n",
     "        self._refresh(key, h, known)\n"
     "        if h > scan.scan_start:\n"
     "            passed = scan.close(h - 1)\n"
     "            if passed:\n"
     "                self.pass_log.append((key, h - 1, [r.key for r in passed]))\n"),
    ("R-F usability strict (a zone whose moment == LOH(h - 1) unusable at h)", OV,
     "        return [z for z in self.view(sub) if z.usable_at <= lim]",
     "        return [z for z in self.view(sub) if z.usable_at < lim]"),
    ("R-G the orchestrator's make_triggers re-emits every earlier var 3 at each trigger (== h -> <= h)", OR,
     "            if int(v3.trigger_event_idx) == int(h):",
     "            if int(v3.trigger_event_idx) <= int(h):"),
    ("R-H the H1 attrs keep the batch var 3 (the final var 3 never written to attrs)", OR,
     '    h1_df.attrs["subsequent_confluence_triggers"] = var3_all\n',
     "    pass\n"),
]
