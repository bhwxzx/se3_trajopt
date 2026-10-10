"""Audit foot Hermite extrema and wheel travel relative to first touchdown."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from check_leg_to_wheel_dynamics import URDF
from visualize_trajectory_csv import load_configurations


def joint_position_violation(model, configurations):
    q = np.asarray(configurations)[:, 7:]
    return float(max(0., np.max(model.lowerPositionLimit[7:] - q),
                     np.max(q - model.upperPositionLimit[7:])))


def hermite_extrema(q0, q1, v0, v1, h):
    """Exact scalar cubic extrema, including endpoints; no temporal sampling."""
    a = 2*q0 - 2*q1 + h*(v0+v1)
    b = -3*q0 + 3*q1 - h*(2*v0+v1)
    c = h*v0
    roots = np.roots(np.trim_zeros([3*a, 2*b, c], 'f'))
    u = [0., 1.] + [float(r.real) for r in roots if abs(r.imag) < 1e-12 and 0 < r.real < 1]
    values = np.array([((a*s+b)*s+c)*s+q0 for s in u])
    return float(values.min()), float(values.max())


def hermite_speed_extrema(q0, q1, v0, v1, h):
    a=2*q0-2*q1+h*(v0+v1);b=-3*q0+3*q1-h*(2*v0+v1)
    u=[0.,1.]
    if a!=0 and 0 < -b/(3*a) < 1:u.append(-b/(3*a))
    values=[(3*a*s*s+2*b*s+h*v0)/h for s in u]
    return float(min(values)),float(max(values))


def recovery_metrics(saved, model, csv_q=None, csv_times=None):
    from nltrajopt.wheel_contact import WheelContactModel
    nodes = saved['solution']['nodes']
    wheels = WheelContactModel.from_metadata(model, saved['info']['wheel_contact_model'])
    times = np.r_[0., np.cumsum([n['dt'] for n in nodes[:-1]])]
    launch = next(k for k,n in enumerate(nodes) if not n['forces'])
    land = next(k for k,n in enumerate(nodes) if all(name in n['forces'] for name in wheels.names))
    feet = {}
    for side in ('left', 'right'):
        name = side + '_foot_joint'
        j = model.joints[model.getJointId(name)]
        q = np.array([n['q'][j.idx_q] for n in nodes])
        v = np.array([n['v'][j.idx_v] for n in nodes])
        extrema = np.array([hermite_extrema(q[k], q[k+1], v[k], v[k+1], nodes[k]['dt']) for k in range(len(nodes)-1)])
        speeds=np.array([hermite_speed_extrema(q[k],q[k+1],v[k],v[k+1],nodes[k]['dt']) for k in range(len(nodes)-1)])
        lo, hi = model.lowerPositionLimit[j.idx_q], model.upperPositionLimit[j.idx_q]
        feet[name] = dict(takeoff_deg=float(np.rad2deg(q[launch])), touchdown_deg=float(np.rad2deg(q[land])),
                          terminal_deg=float(np.rad2deg(q[-1])), node_range_deg=np.rad2deg([q.min(),q.max()]).tolist(),
                          limits_deg=np.rad2deg([lo,hi]).tolist(),
                          hermite_range_deg=np.rad2deg([extrema[:,0].min(),extrema[:,1].max()]).tolist(),
                          hermite_limit_violation_rad=float(max(0.,lo-extrema[:,0].min(),extrema[:,1].max()-hi)))
        feet[name].update(takeoff_velocity_rad_s=float(v[launch]),touchdown_velocity_rad_s=float(v[land]),
                          hermite_max_abs_velocity_rad_s=float(np.max(abs(speeds))),
                          velocity_limit_rad_s=float(model.velocityLimit[j.idx_v]))
        if csv_q is not None:
            vals = np.asarray(csv_q)[:,j.idx_q]
            feet[name]['csv_range_deg'] = np.rad2deg([vals.min(),vals.max()]).tolist()
    positions = np.array([[s['position'][0] for s in wheels.evaluate(np.array(n['q'])).values()] for n in nodes])
    travel = {}
    for i,name in enumerate(wheels.names):
        x = positions[land:,i]; offset = x-x[0]
        travel[name] = dict(touchdown_x_m=float(x[0]), backward_excursion_m=float(max(0.,-offset.min())),
                            forward_excursion_m=float(max(0.,offset.max())), final_net_m=float(offset[-1]),
                            range_m=float(np.ptp(x)), cumulative_path_m=float(np.sum(abs(np.diff(x)))),
                            most_backward_time_s=float(times[land+np.argmin(x)]))
    result = dict(takeoff_time_s=float(times[launch]),touchdown_time_s=float(times[land]),feet=feet,
                  wheel_node_travel=travel, wheel_travel_scope='Optimizer nodes only; dense interval travel is not certified.',
                  max_node_joint_position_violation_rad=joint_position_violation(model,[n['q'] for n in nodes]))
    if csv_q is not None:
        result['max_csv_joint_position_violation_rad'] = joint_position_violation(model,csv_q)
    if csv_q is not None and csv_times is not None:
        if len(csv_times)!=len(csv_q):raise ValueError('CSV time/pose length mismatch')
        mask=np.asarray(csv_times)>=times[land]-1e-9
        csv_positions=np.array([[s['position'][0] for s in wheels.evaluate(np.asarray(q)).values()] for q in np.asarray(csv_q)[mask]])
        result['csv_wheel_backward_excursion_m']={name:float(max(0.,np.max(positions[land,i]-csv_positions[:,i]))) for i,name in enumerate(wheels.names)}
        result['csv_wheel_travel_scope']='Exported frames relative to optimizer touchdown, not a continuous-time bound.'
    return result, times, positions


def audit(solution, csv_path, output):
    saved = json.loads(solution.read_text())
    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    time_file=csv_path.parent/'csv_optimizer_times_s.csv'
    csv_times=np.loadtxt(time_file,skiprows=1) if time_file.exists() else None
    result,times,positions = recovery_metrics(saved,model,load_configurations(csv_path,model),csv_times)
    result['sha256'] = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (solution,csv_path,URDF,Path(__file__))}
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    q = np.array([n['q'] for n in saved['solution']['nodes']])
    foot = q[:,[model.joints[model.getJointId(s+'_foot_joint')].idx_q for s in ('left','right')]]
    np.savetxt(output/'node_motion.csv',np.c_[times,np.rad2deg(foot),positions],delimiter=',',
               header='time_s,left_foot_deg,right_foot_deg,left_wheel_x_m,right_wheel_x_m',comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    axes[0].plot(times,np.rad2deg(foot));axes[0].set_ylabel('Foot angle (deg)')
    axes[0].axhline(0,color='gray',ls=':');axes[0].axhline(110,color='red',ls='--');axes[0].axhline(-110,color='red',ls='--')
    land=np.argmin(abs(times-result['touchdown_time_s']))
    axes[1].plot(times[land:],positions[land:]-positions[land]);axes[1].set_ylabel('Wheel x - touchdown x (m)')
    axes[1].set_xlabel('Time (s)')
    for ax in axes:
        ax.grid();ax.legend(['left','right']);ax.axvspan(result['takeoff_time_s'],result['touchdown_time_s'],alpha=.1)
    fig.tight_layout();fig.savefig(output/'foot_and_retreat.png',dpi=150);plt.close(fig)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('solution',type=Path)
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.argv=[sys.argv[0]]
    print(json.dumps(audit(args.solution,args.csv,args.output),indent=2))
