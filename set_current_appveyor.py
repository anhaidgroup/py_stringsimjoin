import sys

if len(sys.argv) != 2:
    print('Invalid syntax. Correct syntax: python set_current_appveyor.py [test/build]')
    exit(-1)

option = sys.argv[1].lower()

if option not in ['test', 'build']:
    print('Invalid syntax. Correct syntax: python set_current_appveyor.py [test/build]')
    exit(-1)

if option == 'test':
    with open('appveyor_test.yml', 'r') as file_handle:
        app_data = file_handle.read()
else:
    with open('appveyor_conda_build.yml', 'r') as file_handle:                         
        app_data = file_handle.read()     

with open('appveyor.yml', 'w') as file_handle:
    file_handle.write(app_data)    
