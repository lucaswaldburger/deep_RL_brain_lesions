import sys
import argparse

if "./lib" not in sys.path:
    sys.path.append("./lib")

from lib.DQL_testing import *

if __name__== "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(description='Evaluate a model on test set')

    parser.add_argument('-n','--num_episodes', type=int, default=15, help = "Number of episodes that the agent can interact with an image. Default: 15")
    parser.add_argument('-c','--category', type=str, nargs='+', default=None, help='Indicating the categories are going to be used for testing. You can list name of the classes you want to use in testing, for instnce <-c cat dog>. If you wish to use all classes then you can use *. Default: cat')
    parser.add_argument('-m','--model_name', type=str, default='default_model', help='The model name that will be loaded for evaluation. Do not forget to put the model under the path ../experiments/model_name. Default: default_model')
    parser.add_argument('-t','--type',type=str, default=None, help='HGG or LGG my guy')
    args = parser.parse_args()

    f = open("../experiments/{}_{}_experiments/{}/report/evaluate_{}_{}.txt".format(args.category, args.type, args.model_name, args.category[0],type), 'w')
    MAP = []
    for category in args.category:
        print("{} images are being evaluated... \n\n\n\n".format(category))
        MAP.append(DQL_testing(args.num_episodes,
            category,
            args.model_name,
            args.type))
        f.write("Precision for {} category: {}\n".format(category, MAP[-1]))
    
    f.write("MAP over the given category(s): {}\n".format(np.mean(MAP)))
    print("MAP over the given category(s): {}".format(np.mean(MAP)))
    f.close()