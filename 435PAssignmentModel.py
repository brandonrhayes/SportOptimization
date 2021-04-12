###############################################################################
###                        MSCI 435 FINAL TERM PROJECT                      ###
###                          Markus Hamann (20666067)                       ###
###                          Brandon Hayes (20675177)                       ###
###                           Developed 09March2021                         ###
###                            Revised 09Apr2021                            ###
###                          SET PACKING FORMULATION                        ###
###############################################################################

### IMPORTS ###################################################################
import pyomo.environ as pyo  # uses pyomo library for optimization modelling
from pyomo.contrib.sensitivity_toolbox.sens import sipopt
from random import seed
from random import randint
from collections import defaultdict


# FIND SCHEDULES THAT ARE BETTER THAN THE CURRENT ONE
def findSchedule(currentSchedules, val1, val2, possibleSchedules):
    for key in possibleSchedules:
        if ((possibleSchedules[key][0] == val1 and
             possibleSchedules[key][1] == val2) and
                key not in currentSchedules):
            return (key,  [val1, val2])


# HEURISTIC ALGORITHM
def runHeuristic(possibleSchedules, usedSchedules, hideScheduleDetails):
    print("\nHEURISTIC:")
    for scheduleNo in list(usedSchedules.keys()):
        firstValue = (usedSchedules.get(scheduleNo))[0]
        secondValue = (usedSchedules.get(scheduleNo))[1]

        if ((firstValue == 1 or firstValue == 2) and secondValue == 3):
            if not hideScheduleDetails:
                print(
                    "Let's find a schedule where the second value is 4 rather than a 3.")
                print(
                    f"Removing {scheduleNo} with value {usedSchedules.pop(scheduleNo)} from the results.")
                print(
                    f"For reference, here are the currently used schedules {usedSchedules}")
            else:  # still need to remove old schedule
                usedSchedules.pop(scheduleNo)

            result = findSchedule(usedSchedules.keys(), firstValue,
                                  4, possibleSchedules)  # returns key and value
            if not hideScheduleDetails:
                print(
                    f"We found a better schedule to add, adding the result to our schedules: {result}\n")

            usedSchedules[result[0]] = result[1]

    print("\nNew Heuristic Chosen Schedules:\n")
    for scheduleNo in usedSchedules:
        usedSchedules[scheduleNo] = possibleSchedules[scheduleNo]
        print(scheduleNo)
        print(f"{possibleSchedules[scheduleNo]}")
    print("\n\n")

# MAIN ASSIGNMENT ALGORITHM FOR FINDING SCHEDULES


def runAssignmentAlgorithm():
    # LOCAL FUNCTIONS
    # Objective Function
    def obj_func(model):
        return sum(sum(sum(sum(
            model.Ph[h-1] * model.G[t, r, d, h] for t in model.T
        ) for r in model.R) for d in model.D) for h in model.H)

    def eventConstraints(model):

        # Constraint 1
        def rule_const1(model, T, D, H, Q):
            return sum((model.G[T, r, D, H] + model.S[T, r, D, Q]) for r in model.R) <= 10

        model.const1 = pyo.Constraint(
            model.T, model.D, model.H, model.Q, rule=rule_const1)

        # Constraint 2
        def rule_const2(model, T):
            return (model.X[T, T]) == 1

        model.const2 = pyo.Constraint(
            model.T, rule=rule_const2)

        # Constraint 3
        def rule_const3(model, T, R, Q):
            return sum((model.S[T, R, d, Q]) for d in model.D) >= 1

        model.const3 = pyo.Constraint(
            model.T, model.R, model.Q, rule=rule_const3)

        # Constraint 4
        def rule_const4(model, R, D, H):
            return sum(model.G[t, R, D, H] for t in model.T) >= 2

        model.const4 = pyo.Constraint(
            model.R, model.D, model.H, rule=rule_const4)

        # Constraint 5
        def rule_const5(model, T, D, H):
            return sum(model.G[T, r, D, H] for r in model.R) <= 1

        model.const5 = pyo.Constraint(
            model.T, model.D, model.H, rule=rule_const5)

    model = pyo.ConcreteModel()

    # PARAMETERS
    # penalty associated with running games at bad times
    model.Ph = [100, 100, 50, 0]
    model.T = pyo.RangeSet(12)  # set of teams
    model.R = pyo.RangeSet(6)  # set of rinks
    model.D = pyo.RangeSet(6)  # set of days
    model.H = pyo.RangeSet(4)  # set of timeslots for games
    model.Q = pyo.RangeSet(2)  # set of timeslots for practices
    model.P = pyo.RangeSet(3)  # set of pools

    # DECISION VARIABLES
    # If team t from pool p plays a game on rink r on day d at time h
    model.G = pyo.Var(model.T, model.R, model.D,
                      model.H, within=pyo.Binary)

    # If team t from pool p has practice on rink r on day d
    model.S = pyo.Var(model.T, model.R, model.D,
                      model.Q, within=pyo.Binary)

    # If team i from pool p plays team j from pool p
    model.X = pyo.Var(model.T, model.T, within=pyo.Binary)

    # MINIMIZE THE NUMBER OF SCHEDULES
    model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)

    # CALL EVENT CONSTRAINTS AND GENERATE CONSTRAINTS
    eventConstraints(model)

    # Solve second problem now using the generated knapsack options as potential packings
    solver = pyo.SolverFactory('gurobi')
    result = solver.solve(model, tee=False)

    # Prints the results
    print(result)  # usually don't need to print

    print("\nChosen Schedules by Assignment Algorithm:\n")
    G = list(model.G.keys())  # games
    S = list(model.S.keys())  # practice
    M = list(model.X.keys())  # match-ups

    matchupDict = {}
    for i in M:
        if model.X[i]() != 0 and model.X[i]() != None:
            if i[0] not in matchupDict:
                matchupDict[i[0]] = list()
            matchupDict[i[0]].append(i[1])
            print(
                f"Team {i[0]} has been matched with Team {i[1]}")

    gamesDict = {}
    for g in G:
        if model.G[g]() != 0 and model.G[g]() != None:
            if g[0] not in gamesDict:
                gamesDict[g[0]] = list()
            gamesDict[g[0]].append(g[1])
            print(
                f"Team {g[0]} is playing with Team {g[1]}")

    practiceDict = {}
    for p in S:
        if model.S[p]() != 0 and model.S[p]() != None:
            if p[0] not in practiceDict:
                practiceDict[p[0]] = list()
            practiceDict[p[0]].append(p[1])
            print(
                f"Team {p[0]} is practicing with Team {p[1]}")

    return [matchupDict, gamesDict, practiceDict]

# Determines if a schedule can be improved and runs heuristic


def canBeImproved(solutionCanBeImproved, possibleSchedules, usedSchedules, hideScheduleDetails):
    # RUN HEURISTIC IF NECESSARY TO IMPROVE SCHEDULE TO MAXIMIZE FAN VIEWING EXPERIENCE
    if solutionCanBeImproved:
        print("\nHowever, this solution is not optimal and can be improved...")
        runHeuristic(possibleSchedules, usedSchedules, hideScheduleDetails)
    else:
        print("\nThe above solution is optimal and will not be improved by the heuristic.")


def main():
    print("INITIATING MODEL TO GENERATE OPTIMAL GAME SCHEDULE")
    print("In the meanwhile, remember that Brandon & Markus are stellar professionals...\n\n")

    # GENERATE FEASIBLE SCHEDULES
    #feasibleSchedules = runBackpackAlgorithm()

    # RUN SET PACKING ALGORITHM TO FIND FEASIBLE SCHEDULE FOR ROUND ROBIN
    print("HERE IS AN OPTIMAL ROUND ROBIN SCHEDULE")
    result = runAssignmentAlgorithm()  # 6 games, 4 practices
    possibleMatchups = result[0]  # for readibility
    possibleGames = result[1]  # for readibility
    possiblePractices = result[2]  # for readibility
    print(f'Matchups: {possibleMatchups}')
    print(f'Games: {possibleGames}')
    print(f'Practices: {possiblePractices}')


# This ensures that all functions are read before running.
if __name__ == "__main__":
    main()
