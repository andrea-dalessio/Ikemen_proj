from pycode import StudentModel, TeacherModel, IkemenEnvironment
import argparse

parser = argparse.ArgumentParser(description="Select your running mode! --teacherTrain for training with teacher model, --studentTrain for training with student model, --eval for playing with the trained student model")
parser.add_argument('--teacherTrain', action='store_true', help='Run training with teacher model')
parser.add_argument('--studentTrain', action='store_true', help='Run training with student model')
parser.add_argument('--eval', action='store_true', help='Run evaluation with trained student model')

args = parser.parse_args()

# --- Teacher Training Mode ---
if args.teacherTrain:
    env = IkemenEnvironment(training_mode="teacher")
    model = TeacherModel(env)
    model.trainPPO()
    
elif args.studentTrain:
    env = IkemenEnvironment(training_mode="student")
    model = StudentModel(env)
    ...
    
elif args.eval:
    env = IkemenEnvironment(training_mode="student")
    model = StudentModel(env)
    ...

    

