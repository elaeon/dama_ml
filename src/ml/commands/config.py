import os
import subprocess
  
def run(args):
    filepath = os.path.join(os.path.expanduser("~"), ".mlpyp/settings.cfg")
    subprocess.check_call(["nano", filepath])
