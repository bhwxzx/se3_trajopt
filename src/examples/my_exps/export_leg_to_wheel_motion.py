"""Diagnostic q/v-consistent manifold Hermite export; never overwrites a baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from check_leg_to_wheel_dynamics import URDF,JOINT_NAMES
from nltrajopt.hermite_curve import right_jacobian_derivative

METHOD='manifold_cubic_hermite_qv'
CONTACT_METHOD='fixed_foot_manifold_cubic_hermite_qv'


def sample_trajectory(model,nodes,sample_times,fixed_foot_stance=False):
    t=np.r_[0.,np.cumsum([n['dt'] for n in nodes][:-1])]
    qs=np.array([n['q'] for n in nodes]);vs=np.array([n['v'] for n in nodes])
    delta=[pin.difference(model,a,b) for a,b in zip(qs[:-1],qs[1:])]
    end_derivative=[]
    for k,d in enumerate(delta):
        J,_=right_jacobian_derivative(d[:6],np.zeros(6))
        ve=vs[k+1].copy();ve[:6]=np.linalg.solve(J,ve[:6]);end_derivative.append(ve)
    out_q=[];out_v=[]
    if fixed_foot_stance:
        from scipy.spatial.transform import Rotation
        stance_end=t[next(k for k,n in enumerate(nodes) if not n['forces'])]
        foot=next(name for name in nodes[0]['forces'] if name.startswith('left_foot'))
        fid=model.getFrameId(foot);data=model.createData()
        pin.forwardKinematics(model,data,qs[0]);pin.updateFramePlacements(model,data)
        reference=data.oMf[fid].copy()
    for time in sample_times:
        if time<t[0]-1e-9 or time>t[-1]+1e-9:raise ValueError('Sample time lies outside saved trajectory')
        k=int(np.clip(np.searchsorted(t,time,side='right')-1,0,len(t)-2));dt=t[k+1]-t[k]
        s=np.clip((time-t[k])/dt,0.,1.);d=delta[k]
        xi=(-2*s**3+3*s**2)*d+(s**3-2*s**2+s)*dt*vs[k]+(s**3-s**2)*dt*end_derivative[k]
        xidot=((-6*s**2+6*s)*d+(3*s**2-4*s+1)*dt*vs[k]+(3*s**2-2*s)*dt*end_derivative[k])/dt
        J,_=right_jacobian_derivative(xi[:6],np.zeros(6))
        q=pin.integrate(model,qs[k],xi);v=xidot.copy();v[:6]=J@xidot[:6]
        if fixed_foot_stance and time<=stance_end+1e-9:
            # Interpolate independent joint coordinates. Recover dependent base
            # pose/velocity from the same stationary-foot relation as the NLP.
            # This changes between-node base motion and is not dynamic validation.
            pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
            relative=data.oMi[1].inverse()*data.oMf[fid]
            base=reference*relative.inverse()
            q[:3]=base.translation.copy();q[3:7]=Rotation.from_matrix(base.rotation.copy()).as_quat()
            J=pin.computeFrameJacobian(model,data,q,fid,pin.LOCAL)
            v[:6]=np.linalg.solve(J[:,:6],-J[:,6:]@v[6:])
        out_q.append(q);out_v.append(v)
    return np.array(out_q),np.array(out_v)


def csv_values(model,qs):
    ids=[model.joints[model.getJointId(name)].idx_q for name in JOINT_NAMES]
    return np.column_stack((qs[:,:7],qs[:,ids]))


def reconstruction_error(nodes,model,csv_path):
    times=np.loadtxt(csv_path.parent/'csv_optimizer_times_s.csv',skiprows=1)
    method=json.loads((csv_path.parent/'export_metadata.json').read_text())['method']
    if method not in (METHOD,CONTACT_METHOD):raise ValueError('Unknown export method')
    qs,_=sample_trajectory(model,nodes,times,fixed_foot_stance=method==CONTACT_METHOD)
    expected=csv_values(model,qs);actual=np.loadtxt(csv_path,delimiter=',')
    if expected.shape!=actual.shape:return {'matches':False,'reason':'shape mismatch'}
    error=float(np.max(abs(actual-expected)))
    return {'matches':error<=5.01e-7,'max_abs_error':error,'method':method}


def export(solution,output,fps=60.,fixed_foot_stance=False,include_endpoints=False):
    saved=json.loads(solution.read_text());nodes=saved['solution']['nodes']
    model=pin.buildModelFromUrdf(str(URDF),pin.JointModelFreeFlyer())
    duration=sum(n['dt'] for n in nodes[:-1]);times=np.arange(0.,duration,1./fps)
    if not len(times) or times[-1]<duration:times=np.r_[times,duration]
    if fixed_foot_stance and not saved['info'].get('motion_config',{}).get('fixed_foot_integration'):
        raise ValueError('Contact-coordinate export requires a trajectory solved with fixed-foot integration')
    if not include_endpoints:times=times[1:-1]
    qs,vs=sample_trajectory(model,nodes,times,fixed_foot_stance=fixed_foot_stance)
    output.mkdir(parents=True,exist_ok=False)
    path=output/'leg_to_wheel_contact_60hz.csv' if fps==60 else output/f'leg_to_wheel_motion_{fps:g}hz.csv'
    np.savetxt(path,csv_values(model,qs),delimiter=',',fmt='%.6f')
    np.savetxt(output/'csv_optimizer_times_s.csv',times,header='time_s',comments='')
    np.savetxt(output/'analytic_pin_velocities.csv',vs,delimiter=',',header=','.join(['base_vx_local','base_vy_local','base_vz_local','base_wx_local','base_wy_local','base_wz_local']+[str(n)+'_velocity' for n in model.names[2:]]),comments='')
    metadata={'include_endpoints':bool(include_endpoints),'method':CONTACT_METHOD if fixed_foot_stance else METHOD,'fps':fps,'frames':len(times),'source_solution':str(solution.resolve()),
              'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (solution,URDF,Path(__file__))},
              'note':('Stance base pose/velocity reconstructed from fixed left foot and Hermite joint coordinates; right foot checked independently. Intermediate dynamics require independent checks.' if fixed_foot_stance else 'Interpolates q and Pinocchio local v at nodes. No height clipping/projection; intermediate dynamics/contact require independent checks.'),
              'csv_training_ready':False}
    # Check the rounded file itself. Never hide an overshoot by clipping it.
    from audit_foot_recovery import joint_position_violation
    from visualize_trajectory_csv import load_configurations
    violation=joint_position_violation(model,load_configurations(path,model))
    metadata['max_csv_joint_position_violation_rad']=violation
    metadata['csv_joint_position_limits_pass']=violation<=1e-6
    (output/'export_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if fixed_foot_stance:
        raw_q,raw_v=sample_trajectory(model,nodes,times)
        metadata['max_change_from_unconstrained_hermite']={
            'base_translation_m':float(np.max(np.linalg.norm(qs[:,:3]-raw_q[:,:3],axis=1))),
            'generalized_velocity':float(np.max(abs(vs-raw_v)))}
        (output/'export_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('solution',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--fps',type=float,default=60.)
    p.add_argument('--fixed-foot-stance',action='store_true',help='Reconstruct dependent base coordinates during fixed-foot stance')
    p.add_argument("--include-endpoints",action="store_true",help="Include exact initial and terminal samples in diagnostic CSV")
    a=p.parse_args()
    if a.fps<=0:p.error('fps must be positive')
    print(export(a.solution,a.output,a.fps,a.fixed_foot_stance,a.include_endpoints))
