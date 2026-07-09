"""
Download all photos from an Instagram account (default: your own, @taowoosh).

Usage:
    # one-time setup (a .venv-ig/ with instaloader is already created via `uv`;
    # to recreate it: `uv venv .venv-ig && uv pip install -p .venv-ig/bin/python instaloader`).
    PY=.venv-ig/bin/python

    # public posts only, images only (no login):
    $PY scripts/download_instagram.py

    # log in as yourself (needed for private accounts, tagged posts, or when
    # Instagram rate-limits anonymous access — which it usually does):
    $PY scripts/download_instagram.py --login taowoosh

    # include videos too, or point at a different account / folder:
    $PY scripts/download_instagram.py --login taowoosh --include-videos
    $PY scripts/download_instagram.py --target someone_else --dest data/ig

Notes:
- Downloads land (flattened) in data/instagram/<account>/ by default.
- The login session is cached in .venv-ig/../.instaloader-session so you only
  authenticate once. 2FA is supported (you'll be prompted).
- Scraping is against Instagram's ToS and is aggressively rate-limited; this
  targets *your own* content. The fully sanctioned alternative is Instagram's
  "Download your information" export (Settings -> Accounts Center -> Your
  information and permissions -> Download your information).
"""

import argparse
import sys
from pathlib import Path

try:
    import instaloader
except ModuleNotFoundError:
    sys.exit(
        "instaloader is not installed.\n"
        "  python3 -m venv .venv-ig && . .venv-ig/bin/activate && pip install instaloader\n"
        "then re-run this script."
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_DIR = REPO_ROOT / ".instaloader-session"


def build_loader(dest: Path, include_videos: bool) -> "instaloader.Instaloader":
    """An Instaloader configured to save just the media (no json/txt sidecars)."""
    return instaloader.Instaloader(
        dirname_pattern=str(dest),
        # Flat, sortable filenames: 2023-08-14_18-30-00_UTC.jpg
        filename_pattern="{date_utc:%Y-%m-%d_%H-%M-%S}_UTC",
        download_videos=include_videos,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
        compress_json=False,
    )


def authenticate_with_cookie(loader: "instaloader.Instaloader", sessionid: str) -> str:
    """Authenticate using a `sessionid` cookie copied from a logged-in browser.

    Returns the resolved username and caches a normal session so subsequent runs
    don't need the cookie again.
    """
    loader.context._session.cookies.set("sessionid", sessionid.strip(), domain=".instagram.com")
    username = loader.test_login()
    if not username:
        sys.exit("The sessionid is invalid or expired. Grab a fresh one from the browser.")
    loader.context.username = username
    SESSION_DIR.mkdir(exist_ok=True)
    loader.save_session_to_file(filename=str(SESSION_DIR / f"session-{username}"))
    print(f"Authenticated as @{username} via browser cookie.")
    return username


def authenticate(loader: "instaloader.Instaloader", login_user: str) -> None:
    """Reuse a cached session if present, else log in interactively and cache it."""
    SESSION_DIR.mkdir(exist_ok=True)
    session_file = SESSION_DIR / f"session-{login_user}"
    try:
        loader.load_session_from_file(login_user, filename=str(session_file))
        print(f"Reusing cached session for @{login_user}.")
        return
    except FileNotFoundError:
        pass

    import getpass

    password = getpass.getpass(f"Instagram password for @{login_user}: ")
    try:
        loader.login(login_user, password)
    except instaloader.exceptions.TwoFactorAuthRequiredException:
        code = input("Two-factor code: ").strip()
        loader.two_factor_login(code)
    loader.save_session_to_file(filename=str(session_file))
    print(f"Logged in and cached session at {session_file}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download all photos from an Instagram account.")
    parser.add_argument("--target", default="taowoosh", help="Account to download (default: taowoosh).")
    parser.add_argument("--login", metavar="USER", help="Log in as this user before downloading.")
    parser.add_argument(
        "--sessionid-file",
        default=str(SESSION_DIR / "sessionid.txt"),
        help="Path to a file containing an Instagram `sessionid` cookie (default: "
        ".instaloader-session/sessionid.txt). Used if present.",
    )
    parser.add_argument("--include-videos", action="store_true", help="Also download videos.")
    parser.add_argument(
        "--dest",
        default=None,
        help="Output directory (default: data/instagram/<target>).",
    )
    args = parser.parse_args()

    dest = Path(args.dest) if args.dest else REPO_ROOT / "data" / "instagram" / args.target
    dest.mkdir(parents=True, exist_ok=True)

    loader = build_loader(dest, args.include_videos)

    sessionid_file = Path(args.sessionid_file)
    if sessionid_file.is_file() and sessionid_file.read_text().strip():
        authenticate_with_cookie(loader, sessionid_file.read_text())
    elif args.login:
        authenticate(loader, args.login)

    try:
        profile = instaloader.Profile.from_username(loader.context, args.target)
    except instaloader.exceptions.ProfileNotExistsException:
        return f"Profile @{args.target} does not exist or is inaccessible."
    except instaloader.exceptions.LoginRequiredException:
        return "This account requires you to be logged in. Re-run with --login <your_username>."

    total = profile.mediacount
    print(f"@{args.target}: {total} posts. Saving media to {dest}")

    saved = 0
    for i, post in enumerate(profile.get_posts(), start=1):
        try:
            loader.download_post(post, target=Path(dest).name)
            saved += 1
            print(f"  [{i}/{total}] {post.date_utc:%Y-%m-%d} ({'video' if post.is_video else 'image'})")
        except Exception as exc:  # keep going on individual failures
            print(f"  [{i}/{total}] skipped {post.shortcode}: {exc}", file=sys.stderr)

    print(f"Done. Processed {saved} posts into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
