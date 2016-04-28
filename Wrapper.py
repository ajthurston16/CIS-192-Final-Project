import TeamSimulation
import sys
from TeamSimulation import code_to_number

training_set_cache = None
if __name__ == '__main__':
    while True:
        if training_set_cache is None:
            num_cache, code_cache = TeamSimulation.main()
            training_set_cache, target_cache = TeamSimulation.initialize()
        TeamSimulation.run_and_analyze(training_set_cache, target_cache, num_cache, code_cache)
        print "Press enter to re-run the script, CTRL-C to exit"
        sys.stdin.readline()
        reload(TeamSimulation)
    pass

