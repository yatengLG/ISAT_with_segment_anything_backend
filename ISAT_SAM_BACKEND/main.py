# -*- coding: utf-8 -*-
# @Author  : LG

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from ISAT_SAM_BACKEND.utils.model_manage import list_model, download_model, remove_model
from ISAT_SAM_BACKEND.model import sam
from ISAT_SAM_BACKEND.config import BASE_DIR, CHECKPOINT_DIR
import uvicorn
import argparse
from ISAT_SAM_BACKEND.api.v1.routers import api_router
from ISAT_SAM_BACKEND import __version__
import requests

app = FastAPI(title="ISAT-SAM-BACKEND")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

app.include_router(api_router)


def main():
    parser = argparse.ArgumentParser(description="ISAT SAM backend, supporting ISAT use a remote server for SAM encoding.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=False)

    parser.add_argument("--checkpoint", type=str, default="mobile_sam.pt", help="SAM checkpoint name")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Ip")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--workers", type=int, default=1, help="Num of workers")
    parser.add_argument("--dev", action="store_true", help="Reload")

    # 子命令
    parser_model = subparsers.add_parser("model", description="ISAT SAM backend model manage")
    model_group = parser_model.add_mutually_exclusive_group(required=True)      # 互斥组
    model_group.add_argument("--list", action="store_true", help="List model")
    model_group.add_argument("--download", type=str, help="The model name will download")
    model_group.add_argument("--remove", type=str, help="The model name will remove")

    # 子命令
    parser_version = subparsers.add_parser("version", description="ISAT SAM backend version")
    parser_version.add_argument("--latest", action="store_true", help="Check latest version")
    parser_version.add_argument("--current", action="store_true", help="Print current version")
    args = parser.parse_args()

    if args.command is None:
        # init sam
        try:
            model_path = CHECKPOINT_DIR / f'{args.checkpoint}'
            sam.load_model(model_path=str(model_path), use_bfloat16=False)
        except Exception as e:
            print(f"init sam failed: {e}")

        # start server
        uvicorn.run("ISAT_SAM_BACKEND.main:app", host=args.host, port=args.port, workers=args.workers, reload=args.dev)

    elif args.command == "model":
        if args.list:
            list_model(checkpoints_dir=CHECKPOINT_DIR)
        elif args.download:
            download_model(model_name=args.download, checkpoints_dir=CHECKPOINT_DIR)
        elif args.remove:
            remove_model(model_name=args.remove, checkpoints_dir=CHECKPOINT_DIR)

    elif args.command == "version":
        if args.latest:
            print(f"Checking version ...")
            try:
                # check last version
                response = requests.get(f"https://pypi.org/pypi/isat-sam-backend/json", timeout=5)
                latest_version = response.json()["info"]["version"]
                if latest_version != __version__:
                    print(f"The latest version of isat-sam-backend is \033[32m{latest_version}\033[0m")
            except:
                print(f"\033[31mCheck version failed\033[0m")
                pass
        elif args.current:
            print(f"Current version: \033[32m{__version__}\033[0m")
        else:
            print(f"Current version: \033[32m{__version__}\033[0m")
    else:
        pass

if __name__ == '__main__':
    main()
