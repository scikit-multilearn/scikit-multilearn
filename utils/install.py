import pip

def install(package):
    pip.main(['install', package])

def installRequirements(requirements = 'requirements.txt'):
    pip.main(['install', '-r', requirements])

if __name__ == "__main__":
    installRequirements()