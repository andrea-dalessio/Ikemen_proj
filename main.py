from pycode import StudentModel, TeacherModel, IkemenEnvironment, SuperEnvironment
import os
import argparse

parser = argparse.ArgumentParser(prog="main.py", description="Select your running mode!", usage="%(prog)s [options]")
parser.add_argument("-n", help="Chose how many environments to use", action="store")
parser.add_argument('--teacherTrain', action='store_true', help='Run training with teacher model')
parser.add_argument('--studentTrain', action='store_true', help='Run training with student model')
parser.add_argument('--eval', action='store_true', help='Run evaluation with trained student model')

args = parser.parse_args()

dirList = os.listdir()

if 'logs' not in dirList:
    os.mkdir(f"{os.getcwd()}/logs")
    print("Log directory created")
else:
    print("Log dirctory OK")
if 'models_saves' not in dirList:
    os.mkdir(f"{os.getcwd()}/models_saves")
    print("Saves directory created")
else:
    print("Saves dirctory OK")

envNum = args.n if args.n is not None else 4
print("Selected {} environments".format(envNum))
if args.teacherTrain:
    env = SuperEnvironment(training_mode="teacher", environment_number=envNum)
    model = TeacherModel(env, load_checkpoint=True)
    try:
        model.trainPPO()
    finally:
        env.disconnect()
        env.close_game()
    
elif args.studentTrain:
    env = IkemenEnvironment(training_mode="student", port=8080)
    model = StudentModel(env)
    ...
    
elif args.eval:
    env = IkemenEnvironment(training_mode="student", port=8080)
    model = StudentModel(env)
    ...
else:
    parser.print_help()
    

