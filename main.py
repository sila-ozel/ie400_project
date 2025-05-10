import pandas as pd
import ast
from gurobipy import Model, GRB

def part_1(jobs, seekers, location_distances):
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
            type_ok = (si['Desired_Job_Type'] == job['Job_Type'])
            sal_ok = (job['Salary_Range_Max'] >= si['Min_Desired_Salary']) 
            tools_ok = set(job['Required_Skills']).issubset(set(si['Skills']))
            exp_ok = exp_map[si['Experience_Level']] >= exp_map[job['Required_Experience_Level']]
            dist = location_distances.loc[si['Location'], job['Location']]
            loc_ok = (job['Is_Remote'] == 1) or (dist <= si['Max_Commute_Distance'])
            if type_ok and sal_ok and tools_ok and exp_ok and loc_ok:
                C[(i,j)] = 1

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
        assignments = [(i,j) for i in I for j in J if x[i,j].X > 0.5]
        return assignments
    else:
        None

def main():
    jobs = pd.read_csv("./csv_files/jobs.csv")
    seekers = pd.read_csv("./csv_files/seekers.csv")
    location_distances = pd.read_csv("./csv_files/location_distances.csv", index_col=0)

    jobs["Questionnaire"] = jobs["Questionnaire"].apply(ast.literal_eval)
    jobs["Required_Skills"] = jobs["Required_Skills"].apply(ast.literal_eval)
    seekers["Skills"] = seekers["Skills"].apply(ast.literal_eval)
    seekers["Questionnaire"] = seekers["Questionnaire"].apply(ast.literal_eval)
    assignments = part_1(jobs, seekers, location_distances)
    if assignments:
        print("Assignments:")
        for i,j in assignments:
            print(f"Seeker {i} -> Job {j}")
    else:
        print("No optimal solutions found")
    


if __name__ == "__main__":
    main()