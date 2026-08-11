def dedup_videos(video_list):
    seen = {}
    for v in video_list:
        vid = v.get("video_id")
        if vid and vid not in seen:
            seen[vid] = v
    return list(seen.values())


def filter_metadata(records, min_duration, max_duration, title_blacklist):
    title_blacklist = title_blacklist 
    filtered = []
    dropped_no_duration = 0
    dropped_duration_oor = 0
    dropped_title_blacklist = 0

    for r in records:
        title_lower = (r.get("title") or "").lower()
        # if any(bad in title_lower for bad in title_blacklist):
        #     dropped_title_blacklist += 1
        #     print(f"[FILTER] Dropped: {title_lower}")
        #     continue

        matched_bad = next((bad for bad in title_blacklist if bad in title_lower), None)

        if matched_bad:
            dropped_title_blacklist += 1
            # print(f"[FILTER] Dropped: '{title_lower}' (Từ cấm: '{matched_bad}')")
            continue

        dur = r.get("duration_seconds")
        if dur is None:
            dropped_no_duration += 1
            continue  # chưa biết duration -> loại tạm, có thể enrich lại (xem Cell 6)
        if not (min_duration <= dur <= max_duration):
            dropped_duration_oor += 1
            continue
        filtered.append(r)

    print(f"(Bị loại vì thiếu duration: {dropped_no_duration})")
    print(f"(Bị loại vì title chứa từ blacklist: {dropped_title_blacklist})")
    print(f"(Bị loại vì duration ngoài khoảng: {dropped_duration_oor})")
    return filtered