---
name: Push to GitHub before training
description: Always commit and push all changes to GitHub before starting training
type: feedback
---

**Rule:** Trước khi chạy lệnh training (`python tools/train.py`), phải commit và push toàn bộ changes lên GitHub trước.

**Why:** Để đảm bảo code được version control, dễ resume/reproduce training từ bất kỳ machine nào, và không bị mất changes khi có sự cố.

**How to apply:**
1. Kiểm tra `git status` trước khi train
2. Commit tất cả changes: `git add -A && git commit -m "message"`
3. Push lên GitHub: `git push origin main`
4. Sau đó mới chạy training command
