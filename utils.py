#!/usr/bin/env python

import os
import subprocess
import sys
from utils.files import files


class Config:
    root = "."
    meka = "meka"
    skmultilearn = "skmultilearn"
    tests = "tests"
    utils = "utils"


# #######################################################


def test():
    return subprocess.call("py.test", shell=True)


def clean():
    files.removeFilesRecursive(Config.meka, (lambda f: f.endswith(".pyc")))
    files.removeFilesRecursive(Config.skmultilearn, (lambda f: f.endswith(".pyc")))
    files.removeFilesRecursive(Config.tests, (lambda f: f.endswith(".pyc")))
    files.removeFilesRecursive(Config.utils, (lambda f: f.endswith(".pyc")))


def package():
    sourceDirs = [Config.meka, Config.skmultilearn, Config.tests, Config.utils]
    for sourceDir in sourceDirs:
        for root, dirs, files in os.walk(sourceDir):
            for dir in dirs:
                initFile = os.path.join(root, dir, "__init__.py")
                if not os.path.isfile(initFile):
                    print("creating : %s" % initFile)
                    open(initFile, "a").close()


def lint(full=False):
    from pylint import epylint

    sources = [Config.root, Config.meka, Config.skmultilearn]
    if full:
        fullReport = "y"
    else:
        fullReport = "n"

    config = '--rcfile ./utils/pylint.config --msg-template="{C}:{msg_id}:{line:3d},{column:2d}:{msg}({symbol})" -r %s %s'
    for dir in sources:
        print("lint %s" % dir)
        epylint.py_run(config % (fullReport, dir), script="pylint")


def lint_full():
    lint(True)


def default():
    clean()
    package()
    lint()
    # test()


def install():
    try:
        import build.install as installer
    except ImportError:
        subprocess.call("python ./build/get-pip.py", shell=True)
        import utils.install as installer

    installer.installRequirements("./build/requirements.txt")


########################################################


def step(msg):
    span = "=" * ((80 - len(msg)) / 2)
    print(" ".join([span, msg, span]))


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.realpath(__file__))

    if len(sys.argv) > 1:
        for task in sys.argv[1:]:
            if task in locals():
                step(task)
                locals()[task]()
            else:
                print('Error: task "%s" not found' % task)
                sys.exit(1)
    else:
        default()
