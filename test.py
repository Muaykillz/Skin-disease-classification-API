import re

with open('requirements.txt', 'r') as file:
    requirements = file.readlines()

required_libraries = set()
for line in requirements:
    match = re.match(r'^(\w+)==', line)
    if match:
        required_libraries.add(match.group(1))

unused_libraries = set()
with open('requirements.txt', 'w') as file:
    for line in requirements:
        match = re.match(r'^(\w+)==', line)
        if match and match.group(1) not in required_libraries:
            unused_libraries.add(match.group(1))
        else:
            file.write(line)

print("Unused libraries:", unused_libraries)
