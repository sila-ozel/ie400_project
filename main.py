import pandas as pd
import ast
from gurobipy import Model, GRB
import matplotlib.pyplot as plt

def part_1(jobs, I, J, C):
    model = Model('job_matching')
    x = model.addVars(I, J, vtype=GRB.BINARY, name='x')
    # target is the objective function of this integer programming problem
    target = sum(jobs.loc[j,'Priority_Weight'] * x[i,j] for i in I for j in J)
    model.setObjective(target, GRB.MAXIMIZE)
    # add constraints: each seeker must be assigned to one job, number of assignments should be less than or equal to the job capacity,
    # and the compatibility constraints of the seeker and job must be satisfied
    model.addConstrs((sum(x[i,j] for j in J) <= 1 for i in I), name='one_job_per_seeker')
    model.addConstrs((sum(x[i,j] for i in I) <= jobs.loc[j,'Num_Positions'] for j in J), name='job_capacity')
    model.addConstrs((x[i,j] <= C[i,j] for i in I for j in J), name='compatibility')


    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Maximum weighted priority sum: {model.objVal}")
        assignments = []
        for i in I:
            for j in J:
                if x[i, j].X == 1:
                    assignments.append((i,j))
        return (assignments, model.ObjVal)
    else:
        return None
    
def part_2(w, jobs, d_max, d_ij_pairs, I, J, C, Mw):
    model = Model(f'min_max_dissim_{int(w*100)}')
    x2 = model.addVars(I, J, vtype=GRB.BINARY, name='x2')
    z  = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=5, name='z')

    # one job per seeker, job capacity and compatibility constraints are also included here
    model.addConstrs((sum(x2[i,j] for j in J) <= 1 for i in I), name='one_job_per_seeker')
    model.addConstrs((sum(x2[i,j] for i in I) <= jobs.loc[j,'Num_Positions'] for j in J), name='job_capacity')
    model.addConstrs((x2[i,j] <= C[i,j] for i in I for j in J), name='compatibility')
    # Priority weight threshold
    model.addConstr(sum(jobs.loc[j,'Priority_Weight'] * x2[i,j] for i in I for j in J) >= w * Mw, name='min_priority')
    # Max-dissimilarity definition
    model.addConstrs((z >= d_max[(i,j)] * x2[i,j] for (i,j) in d_ij_pairs), name='max_diss')
    # Objective function
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        assignments = []
        for i in I:
            for j in J:
                if x2[i,j].X == 1:
                    assignments.append((i,j))
        
        return (assignments, model.objVal)
    else:
        return None


def main():
    jobs = pd.read_csv("./csv_files/jobs.csv")
    seekers = pd.read_csv("./csv_files/seekers.csv")
    location_distances = pd.read_csv("./csv_files/location_distances.csv", index_col=0)

    jobs["Questionnaire"] = jobs["Questionnaire"].apply(ast.literal_eval)
    jobs["Required_Skills"] = jobs["Required_Skills"].apply(ast.literal_eval)
    seekers["Skills"] = seekers["Skills"].apply(ast.literal_eval)
    seekers["Questionnaire"] = seekers["Questionnaire"].apply(ast.literal_eval)
    exp_map = {'Entry-level': 1, 'Mid-level': 2, 'Senior': 3, 'Lead': 4, 'Manager': 5}

    I = seekers['Seeker_ID'].tolist()
    J = jobs['Job_ID'].tolist()

    seekers = seekers.set_index('Seeker_ID')
    jobs = jobs.set_index("Job_ID")

    C = {(i,j): 0 for i in I for j in J}
    for i in I:
        si = seekers.loc[i]
        for j in J:
            job = jobs.loc[j]
            # the seeker's desired job type should match the job's job type.
            type_ok = (si['Desired_Job_Type'] == job['Job_Type'])
            # we thought that if the upper bound of the salary is greater than or equal to the minimum desired salary of the seeker,
            # then the salary offer might be suitable for the seeker and the maximum number of job openings can be filled (given that this is a more relaxed constraint)
            sal_ok = (job['Salary_Range_Max'] >= si['Min_Desired_Salary']) 
            # we also tried with the following salary constraint, which was a more restrictive constraint.
            # sal_ok = (job['Salary_Range_Min'] >= si['Min_Desired_Salary']) 
            # the job's required skills must be a subset of the seeker's skills.
            tools_ok = set(job['Required_Skills']).issubset(set(si['Skills']))
            # the experience level of the seeker should be greater than or equal to the required experience level of the job. (see the exp_map for the mapping)
            exp_ok = exp_map[si['Experience_Level']] >= exp_map[job['Required_Experience_Level']]
            # for the location constraint, either the job should be remote or within the max commute distance of the seeker.
            dist = location_distances.loc[si['Location'], job['Location']]
            loc_ok = (job['Is_Remote'] == 1) or (dist <= si['Max_Commute_Distance'])
            if type_ok and sal_ok and tools_ok and exp_ok and loc_ok:
                C[(i,j)] = 1
    (assignments, Mw) = part_1(jobs, I, J, C)
    print(f"Optimal solution = {Mw}")
    # we wanted to see the job assignments, comment out the below code snippet if you want to observe as well
    # if assignments:
    #     print("Assignments:")
    #     for i,j in assignments:
    #         print(f"Seeker {i} -> Job {j}")
    # else:
    #     print("No optimal solutions found")


    # we calculated the dissimilarity score for the questionnaire answers
    d_max = {}  # store dij for each pair
    d_ij_pairs = []
    for i in I:
        qi = seekers.loc[i,'Questionnaire']
        for j in J:
            if C[(i,j)]:
                qj = jobs.loc[j,'Questionnaire']
                dij = sum(abs(qi[k] - qj[k]) for k in range(len(qi))) / len(qi)
                d_max[(i,j)] = dij
                d_ij_pairs.append((i,j))
    # we also tried with different values of ws
    w_vals = [70, 75, 80, 85, 90, 95, 100]
    opt_vals = {} # this is used for storing the optimal solutions for the above w values.
    for w in w_vals:
        (assignments_part_2, part_2_obj_val) = part_2(w/100, jobs, d_max, d_ij_pairs, I, J, C, Mw)
        if part_2_obj_val:
            opt_vals[w] = part_2_obj_val
            print(f"Second objective value: {part_2_obj_val}")
    # we used opt_vals.keys() for the weights because some w values may not have optimal solutions, in that case we won't include it in the graph.
    weights = list(opt_vals.keys())
    objective_values = list(opt_vals.values())
    print(f"Mw: {Mw}, oprimal values for w = [70, 75, 80, 85, 90, 95, 100]: {objective_values}")

    # plot the objective solutions - ws graph
    plt.figure(figsize=(10, 6))
    plt.plot(weights, objective_values, marker='o', linestyle='-', color='b', label='Objective Value')
    plt.title('Objective Values vs. Weight Thresholds')
    plt.xlabel('Weight Threshold (%)')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # again to see the assignments of the second part, comment out the code below.
    # if assignments_part_2:
    #     print("Assignments:")
    #     for i,j in assignments_part_2:
    #         print(f"Seeker {i} -> Job {j}")
    # else:
    #     print("No optimal solutions found")
    


if __name__ == "__main__":
    main()