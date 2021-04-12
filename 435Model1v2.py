###############################################################################
###                        MSCI 435 FINAL TERM PROJECT                      ###
###                          Markus Hamann (20666067)                       ###
###                          Brandon Hayes (20675177)                       ###
###                           Developed 26March2021                         ###
###                                                                         ###
###                                                                         ###
###############################################################################

### IMPORTS ###################################################################
# uses pyomo library for optimization modelling
import pyomo.environ as pyo
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


# GENERATE FEASIBLE SOLUTIONS USING THE BACKPACK ALGORITHM
def runBackpackAlgorithm():
    # LOCAL FUNCTIONS
    # Objective Function
    def obj_func(model):
        return sum(model.X[t]*model.w[t] for t in model.T)

    # Constraint Generation
    def humanContactConstraint(model, T):
        # Human Contact Limit
        V = 20
        return sum(model.X[t]*model.c[t] for t in model.T) <= V

    schedules = []  # array of schedules, _ is an invisible variable for iterations
    print("Please be patient while possible schedules are generated...")
    for _ in range(1, 100):
        # generate some integers for weighting
        value1 = randint(0, 10000)
        value2 = randint(0, 10000)
        value3 = randint(0, 10000)
        value4 = randint(0, 10000)

        seed(value1)

        factors = {1: 7, 2: 7, 3: 10, 4: 10}
        weight = {1: value1, 2: value2, 3: value3, 4: value4}

        # Model intialization
        model = pyo.ConcreteModel()

        # Indexes for timeslots t
        # Number of time slots
        t = 4
        model.T = pyo.RangeSet(t)

        # Parameter variable xt
        model.X = pyo.Var(model.T, within=pyo.Binary)

        # Item sizes matrix ct
        model.c = pyo.Param(
            model.T, initialize=factors, default=0)

        # Item weights matrix
        model.w = pyo.Param(
            model.T, initialize=weight, default=0)

        model.objective = pyo.Objective(rule=obj_func, sense=pyo.maximize)

        model.const1 = pyo.Constraint(model.T, rule=humanContactConstraint)

        # Solves knapsack using Gurobi
        solver = pyo.SolverFactory('gurobi')
        solver.solve(model, tee=False)

        options = []
        for i in list(model.X.keys()):
            if model.X[i]() != 0:
                options.append(i)

        schedules.append(options)

    return schedules  # These are the feabile schedules used later on


# This sub finds and prints the number of each events in a schedule (game or practice)
def findNumberOfEvents(possibleSchedules, hideScheduleDetails):
    # Create empty dictionary to store feasible packings by item type
    Games = {}
    Practices = {}

    # Load in items to empty dictionary
    for key in possibleSchedules.keys():
        scheduledDays = possibleSchedules[key]
        if sum(scheduledDays) == 3:  # All Practices
            Games[key] = (0)
            Practices[key] = (2)
        elif (sum(scheduledDays) >= 4 and sum(scheduledDays) <= 6):  # 1 game 1 pracy
            Games[key] = (1)
            Practices[key] = (1)
        elif (sum(scheduledDays) >= 7):  # All Games
            Games[key] = (2)
            Practices[key] = (0)

    if not hideScheduleDetails:
        print("The possible schedules are...")
        print(f"{possibleSchedules}\n")
        print(f"The schedules have x games...")
        print(f"{Games}\n")
        print(f"The schedules have x practices...")
        print(f"{Practices}\n")

    # return games and practices for use
    return (Games, Practices)


# MAIN SET PACKING ALGORITHM FOR FINDING SCHEDULES
def runSetPackingAlgorithm(schedules, gameConst, practiceConst, hideScheduleDetails):
    # LOCAL FUNCTIONS
    # Objective Function
    def obj_func(model):
        return sum(model.A[h] for h in possibleSchedules.keys())

    def eventConstraints(gameConst, practiceConst):
        # Iterativly add constraints by item, t = timeslot
        model.iterative_constraint = pyo.ConstraintList()
        gameLHS = 0
        for t in Games:
            gameLHS = gameLHS + Games[t]*model.A[t]

        model.iterative_constraint.add(gameLHS == gameConst)

        # Iterativly add constraints by item, t = timeslot
        practiceLHS = 0
        for t in Practices:
            practiceLHS = practiceLHS + Practices[t]*model.A[t]

        model.iterative_constraint.add(practiceLHS == practiceConst)

    # Arrange into dictionary with unique identifier for each packing
    possibleSchedules = {'Schedule_' + str(i): schedules[i]
                         for i in range(1, len(schedules))}

    numEvents = findNumberOfEvents(possibleSchedules, hideScheduleDetails)
    Games = numEvents[0]
    Practices = numEvents[1]

    model = pyo.ConcreteModel()

    # Indexes for Number of Schedules
    model.H = pyo.RangeSet(len(schedules))

    # Indexes for Timeslots Available in Day
    model.L = pyo.RangeSet(4)

    # Decision variable alpha h
    model.A = pyo.Var(possibleSchedules.keys(), within=pyo.Binary)

    # MINIMIZE THE NUMBER OF SCHEDULES
    model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)

    # CALL EVENT CONSTRAINTS AND
    # GENERATE CONSTRAINTS FOR REQUIRED GAMES AND PRACITCES
    eventConstraints(gameConst, practiceConst)

    # Solve second problem now using the generated knapsack options as potential packings
    solver = pyo.SolverFactory('gurobi')
    result = solver.solve(model, tee=False)

    # Prints the results
    print(result)  # usually don't need to print

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

# Determines if a schedule can be improved and runs heuristic


def canBeImproved(solutionCanBeImproved, possibleSchedules, usedSchedules, hideScheduleDetails):
    # RUN HEURISTIC IF NECESSARY TO IMPROVE SCHEDULE TO MAXIMIZE FAN VIEWING EXPERIENCE
    if solutionCanBeImproved:
        print("\nHowever, this solution is not optimal and can be improved...")
        runHeuristic(possibleSchedules, usedSchedules, hideScheduleDetails)
    else:
        print("\nThe above solution is optimal and will not be improved by the heuristic.")


def main():
    hideScheduleDetails = True  # used to hide additional print statements
    print("INITIATING MODEL TO GENERATE OPTIMAL GAME SCHEDULE")
    print("In the meanwhile, remember that Brandon & Markus are stellar professionals...\n\n")

    # GENERATE FEASIBLE SCHEDULES
    feasibleSchedules = runBackpackAlgorithm()

    # RUN SET PACKING ALGORITHM TO FIND FEASIBLE SCHEDULE FOR ROUND ROBIN
    print("HERE IS THE OPTIMAL ROUND ROBIN SCHEDULE")
    result = runSetPackingAlgorithm(
        feasibleSchedules, 768, 512, hideScheduleDetails)  # 6 games, 4 practices
    possibleSchedules = result[0]  # for readibility
    usedSchedules = result[1]  # for readibility
    solutionCanBeImproved = result[2]  # for readibility

    # improve schedule if can be improved
    canBeImproved(solutionCanBeImproved, possibleSchedules,
                  usedSchedules, hideScheduleDetails)

    # RUN SET PACKING ALGORITHM TO FIND FEASIBLE SCHEDULE FOR PLAYOFF ROUND 1
    print("HERE IS THE OPTIMAL ROUND 1 PLAYOFF SCHEDULE")
    result = runSetPackingAlgorithm(
        feasibleSchedules, 2, 2, hideScheduleDetails)  # 2 games, 2 practices
    possibleSchedules = result[0]  # for readibility
    usedSchedules = result[1]  # for readibility
    solutionCanBeImproved = result[2]  # for readibility

    # improve schedule if can be improved
    canBeImproved(solutionCanBeImproved, possibleSchedules,
                  usedSchedules, hideScheduleDetails)

    # RUN SET PACKING ALGORITHM TO FIND FEASIBLE SCHEDULE FOR PLAYOFF ROUND 2
    print("HERE IS THE OPTIMAL ROUND 2 PLAYOFF SCHEDULE")
    result = runSetPackingAlgorithm(
        feasibleSchedules, 2, 2, hideScheduleDetails)  # 2 games, 2 practices
    possibleSchedules = result[0]  # for readibility
    usedSchedules = result[1]  # for readibility
    solutionCanBeImproved = result[2]  # for readibility

    # improve schedule if can be improved
    canBeImproved(solutionCanBeImproved, possibleSchedules,
                  usedSchedules, hideScheduleDetails)

    # RUN SET PACKING ALGORITHM TO FIND FEASIBLE SCHEDULE FOR PLAYOFF ROUND 3
    print("HERE IS THE OPTIMAL ROUND 3 PLAYOFF SCHEDULE")
    result = runSetPackingAlgorithm(
        feasibleSchedules, 2, 2, hideScheduleDetails)  # 2 games, 2 practices
    possibleSchedules = result[0]  # for readibility
    usedSchedules = result[1]  # for readibility
    solutionCanBeImproved = result[2]  # for readibility

    # improve schedule if can be improved
    canBeImproved(solutionCanBeImproved, possibleSchedules,
                  usedSchedules, hideScheduleDetails)


# This ensures that all functions are read before running.
if __name__ == "__main__":
    main()
