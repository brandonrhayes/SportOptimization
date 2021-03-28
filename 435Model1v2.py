# Markus Hamann (20666067)
# Brandon Hayes (20675177)
# MSCI 435 Term Project

### IMPORTS #######################################
# uses pyomo library for optimization modelling
import pyomo.environ as pyo
from random import seed
from random import randint
from collections import defaultdict

# FIND SCHEDULES THAT ARE BETTER THAN THE CURRENT ONE


def findSchedule(currentSchedules, val1, val2, possibleSchedules):
    for key in possibleSchedules:
        if ((possibleSchedules[key][0] == val1 and possibleSchedules[key][1] == val2) and key not in currentSchedules):
            return (key,  [val1, val2])

# HEURISTIC ALGORITHM


def runHeuristic(possibleSchedules, usedSchedules):
    print("\nHEURISTIC:")
    for scheduleNo in list(usedSchedules.keys()):
        firstValue = (usedSchedules.get(scheduleNo))[0]
        secondValue = (usedSchedules.get(scheduleNo))[1]

        if ((firstValue == 1 or firstValue == 2) and secondValue == 3):
            print("Let's find a schedule where the second value is 4 rather than a 3.")
            print(
                f"Removing {scheduleNo} with value {usedSchedules.pop(scheduleNo)} from the results.")
            print(
                f"For reference, here are the currently used schedules {usedSchedules}")
            result = findSchedule(usedSchedules.keys(), firstValue,
                                  4, possibleSchedules)  # returns key and value
            print(
                f"We found a better schedule to add, adding the result to our schedules: {result}\n")
            usedSchedules[result[0]] = result[1]

    print("\nNew Chosen Schedules:\n")
    for scheduleNo in usedSchedules:
        usedSchedules[scheduleNo] = possibleSchedules[scheduleNo]
        print(scheduleNo)
        print(possibleSchedules[scheduleNo])


# GENERATE FEASIBLE SOLUTIONS USING THE BACKPACK ALGORITHM
def runBackpackAlgorithm():
    schedules = []
    for p in range(1, 30):
        # generate some integers for weighting
        value1 = randint(0, 10000)
        value2 = randint(0, 10000)
        value3 = randint(0, 10000)
        value4 = randint(0, 10000)

        seed(value1)

        factors = {1: 7, 2: 7, 3: 10, 4: 10}
        weight = {1: value1, 2: value2, 3: value3, 4: value4}

        # Number of time slots
        t = 4

        # Human Contact Limit
        V = 20

        # Model intialization
        model = pyo.ConcreteModel()

        # Indexes for time slots t
        model.T = pyo.RangeSet(t)

        # Parameter variable xt
        model.X = pyo.Var(model.T, within=pyo.Binary)

        # Item sizes matrix ct
        model.c = pyo.Param(
            model.T, initialize=factors, default=0)

        # Item weights matrix
        model.w = pyo.Param(
            model.T, initialize=weight, default=0)

        # objective function
        def obj_func(model):
            return sum(model.X[t]*model.w[t] for t in model.T)

        model.objective = pyo.Objective(rule=obj_func, sense=pyo.maximize)

        def rule_const1(model, T):
            return sum(model.X[t]*model.c[t] for t in model.T) <= V

        model.const1 = pyo.Constraint(model.T, rule=rule_const1)

        # Solves knapsack using Gurobi
        solver = pyo.SolverFactory('gurobi')
        solver.solve(model, tee=False)

        l = list(model.X.keys())

        options = []
        for i in l:
            if model.X[i]() != 0:
                options.append(i)

        schedules.append(options)

    return schedules

# MAIN SET PACKING ALGORITHM FOR FINDING SCHEDULES


def runSetPackingAlgorithm(schedules):
    # Arrange into dictionary with unique identifier for each packing
    possibleSchedules = {'Schedule_' + str(i): schedules[i]
                         for i in range(1, len(schedules))}

    # Create empty dictionary to store feasible packings by item type

    Games = {}
    Practices = {}

    # Load in items to empty dictionary
    i = 1
    for key in possibleSchedules.keys():
        scheduledDays = possibleSchedules[key]
        if sum(scheduledDays) == 3:  # All Practices
            Games['Schedule_' + str(i)] = (0)
            Practices['Schedule_' + str(i)] = (2)
        elif (sum(scheduledDays) >= 4 and sum(scheduledDays) <= 6):  # 1 game 1 pracy
            Games['Schedule_' + str(i)] = (1)
            Practices['Schedule_' + str(i)] = (1)
        elif (sum(scheduledDays) >= 7):  # All Games
            Games['Schedule_' + str(i)] = (2)
            Practices['Schedule_' + str(i)] = (0)
        i = i + 1

    print("The possible schedules are...")
    print(f"{possibleSchedules}\n")
    print(f"The schedules have x games...")
    print(f"{Games}\n")
    print(f"The schedules have x practices...")
    print(f"{Practices}\n")

    h = len(schedules)
    l = 4

    model = pyo.ConcreteModel()

    # Indexes for Number of Schedules
    model.H = pyo.RangeSet(h)

    # Indexes for Timeslots Available in Day
    model.L = pyo.RangeSet(l)

    # Decision variable alpha h
    model.A = pyo.Var(possibleSchedules.keys(), within=pyo.Binary)

    # MINIMIZE THE NUMBER OF SCHEDULES

    def obj_func(model):
        return sum(model.A[h] for h in possibleSchedules.keys())

    model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)

    # Iterativly add constraints by item, t = timeslot
    model.iterative_constraint = pyo.ConstraintList()
    gameLHS = 0
    for t in Games:
        gameLHS = gameLHS + Games[t]*model.A[t]

    model.iterative_constraint.add(gameLHS == 6)

    # Iterativly add constraints by item, t = timeslot
    practiceLHS = 0
    for t in Practices:
        practiceLHS = practiceLHS + Practices[t]*model.A[t]

    model.iterative_constraint.add(practiceLHS == 4)

    # Solve second problem now using the generated knapsack options as potential packings
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model, tee=False)

    # Prints the results
    # print(result)

    print("\nChosen Schedules by Set Packing Algorithm:\n")
    l = list(model.A.keys())
    usedSchedules = {}
    solutionCanBeImproved = False
    for scheduleNo in l:
        if model.A[scheduleNo]() != 0:
            usedSchedules[scheduleNo] = possibleSchedules[scheduleNo]
            print(scheduleNo)
            print(possibleSchedules[scheduleNo])
            firstValue = (usedSchedules.get(scheduleNo))[0]
            secondValue = (usedSchedules.get(scheduleNo))[1]
            if ((firstValue == 1 or firstValue == 2) and secondValue == 3):
                solutionCanBeImproved = True

    return [possibleSchedules, usedSchedules, solutionCanBeImproved]


def main():
    print("INITIATING MODEL")
    print("In the meanwhile, remember that Brandon & Markus are stellar professionals...\n\n")

    # GENERATE FEASIBLE SCHEDULES
    feasibleSchedules = runBackpackAlgorithm()

    # RUN SET PACKING ALGORITHM TO FIND FEASIBLE SCHEDULE
    result = runSetPackingAlgorithm(feasibleSchedules)
    possibleSchedules = result[0]
    usedSchedules = result[1]
    solutionCanBeImproved = result[2]

    # RUN HEURISTIC IF NECESSARY TO IMPROVE SCHEDULE TO MAXIMIZE FAN VIEWING EXPERIENCE
    if solutionCanBeImproved:
        print("\nHowever, this solution is not optimal and can be improved...")
        runHeuristic(possibleSchedules, usedSchedules)
    else:
        print("\nThe above solution is optimal and will not be improved by the heuristic.")


if __name__ == "__main__":
    main()
