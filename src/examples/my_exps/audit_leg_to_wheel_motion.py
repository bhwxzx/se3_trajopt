"""Motion-intent audit: hip zero, jump height, touchdown placement, flight momentum."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from lw_validation_helpers import write_json
from check_leg_to_wheel_dynamics import URDF


def audit_motion(solution, output):
    from nltrajopt.wheel_contact import WheelContactModel
    saved=json.loads(solution.read_text());nodes=saved['solution']['nodes']
    model=pin.buildModelFromUrdf(str(URDF),pin.JointModelFreeFlyer());data=model.createData()
    wheels=WheelContactModel.from_metadata(model,saved['info']['wheel_contact_model'])
    times=np.r_[0.,np.cumsum([n['dt'] for n in nodes][:-1])]
    takeoff=next(k for k,n in enumerate(nodes) if not n['forces'])
    landing=next(k for k,n in enumerate(nodes) if all(name in n['forces'] for name in wheels.names))
    rows=[]
    for t,n in zip(times,nodes):
        q,v,a=[np.array(n[key]) for key in ('q','v','a')]
        com=pin.centerOfMass(model,data,q,v,a).copy();vc=data.vcom[0].copy();ac=data.acom[0].copy()
        momentum=pin.computeCentroidalMomentum(model,data,q,v).vector.copy()
        state=wheels.evaluate(q,v);p=np.mean([s['position'] for s in state.values()],axis=0)
        r={'time_s':float(t),'flight':int(not n['forces'])}
        for key,vec in [('base',q[:3]),('com',com),('com_v',vc),('com_a',ac),('wheel_mid',p),('angular_momentum',momentum[3:])]:
            for ax,value in zip('xyz',vec):r[key+'_'+ax]=float(value)
        r['com_minus_wheel_x_m']=float(com[0]-p[0]);r['base_minus_wheel_x_m']=float(q[0]-p[0])
        for side in ('left','right'):
            j=model.joints[model.getJointId(side+'_hip_joint')]
            r[side+'_hip_deg']=float(np.rad2deg(q[j.idx_q]));r[side+'_hip_v_rad_s']=float(v[j.idx_v]);r[side+'_hip_a_rad_s2']=float(a[j.idx_v])
        rows.append(r)
    flight=rows[takeoff:landing+1]
    apex=max(flight,key=lambda r:r['com_z'])
    p0=np.array([rows[takeoff]['com_'+ax] for ax in 'xyz']);v0=np.array([rows[takeoff]['com_v_'+ax] for ax in 'xyz'])
    h0=np.array([rows[takeoff]['angular_momentum_'+ax] for ax in 'xyz'])
    drifts=[];pos_errors=[];angular=[]
    for r in flight:
        dt=r['time_s']-times[takeoff]
        drifts.append(np.array([r['com_v_'+ax] for ax in 'xyz'])-v0-model.gravity.linear*dt)
        pos_errors.append(np.array([r['com_'+ax] for ax in 'xyz'])-p0-v0*dt-.5*model.gravity.linear*dt**2)
        angular.append(np.array([r['angular_momentum_'+ax] for ax in 'xyz'])-h0)
    drift=np.array(drifts)
    summary={'solution':str(solution.resolve()),'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (solution,URDF,Path(__file__))},
             'takeoff_time_s':float(times[takeoff]),'landing_time_s':float(times[landing]),'flight_duration_s':float(times[landing]-times[takeoff]),
             'initial_com_z_m':rows[0]['com_z'],'flight_com_apex_z_m':apex['com_z'],'flight_com_apex_time_s':apex['time_s'],
             'flight_com_apex_above_initial_m':apex['com_z']-rows[0]['com_z'],
             'touchdown':rows[landing], 'final_wheel_mid_x_m':rows[-1]['wheel_mid_x'],
             'touchdown_wheel_minus_final_x_m':rows[landing]['wheel_mid_x']-rows[-1]['wheel_mid_x'],
             'max_flight_ballistic_velocity_drift_m_s':np.max(abs(drift),axis=0).tolist(),
             'touchdown_ballistic_velocity_drift_m_s':drift[-1].tolist(),
             'max_flight_ballistic_position_error_m':np.max(np.abs(pos_errors),axis=0).tolist(),
             'max_flight_angular_momentum_drift_kg_m2_s':np.max(np.abs(angular),axis=0).tolist(),
             'hip':{side:{'max_abs_angle_deg':max(abs(r[side+'_hip_deg']) for r in rows),
                          'max_abs_velocity_rad_s':max(abs(r[side+'_hip_v_rad_s']) for r in rows),
                          'max_abs_acceleration_rad_s2':max(abs(r[side+'_hip_a_rad_s2']) for r in rows)} for side in ('left','right')},
             'note':'Flight includes touchdown state before any modeled impulse (none present). Ballistic drift tests interval consistency, not only instantaneous RNEA.'}
    output.mkdir(parents=True,exist_ok=False)
    with (output/'motion_nodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for side in ('left','right'):
        axes[0,0].plot(times,[r[side+'_hip_deg'] for r in rows],label=side)
        axes[0,1].plot(times,[r[side+'_hip_v_rad_s'] for r in rows],label=side)
    for key in ('base_z','com_z','wheel_mid_z'):axes[1,0].plot(times,[r[key] for r in rows],label=key)
    axes[1,0].axhline(rows[0]['com_z'],ls=':',color='gray',label='initial COM z')
    for key in ('base_x','com_x','wheel_mid_x'):axes[1,1].plot(times,[r[key] for r in rows],label=key)
    axes[1,1].axhline(rows[-1]['wheel_mid_x'],ls=':',color='gray',label='final wheel x')
    for j,ax in enumerate('xyz'):
        axes[2,0].plot(times[takeoff:landing+1],drift[:,j],label=ax)
        axes[2,1].plot(times[takeoff:landing+1],np.array(pos_errors)[:,j],label=ax)
    for ax,title in zip(axes.flat,['Hip angle (deg)','Hip velocity (rad/s)','World height (m)','World x (m)',
                                  'Flight COM velocity drift (m/s)','Flight COM position error (m)']):
        ax.set_title(title);ax.grid(alpha=.3);ax.legend();ax.axvline(times[landing],ls=':',color='black')
    fig.tight_layout();fig.savefig(output/'motion.png',dpi=160);plt.close(fig)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('solution',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.argv=[sys.argv[0]]
    print(json.dumps(audit_motion(args.solution,args.output),indent=2))
