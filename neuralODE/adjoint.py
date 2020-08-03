import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

def odeint(z0,func,t0,t1,method="RK"):
    n_steps = 10
    dt = (t1.item()-t0.item())/n_steps
    t = t0.clone()#cloneしないとt0も更新されてしまう
    z = z0
    
    if method=="Euler":
        for _ in range(n_steps):
            f = func(z,t)
            if type(f) is tuple:
                for i in range(len(f)):
                    z[i] += dt*f[i]
            else:
                z += dt * f
            t += dt

    elif method=="RK":
        for _ in range(n_steps):
            k1 = func(z,t)
            z_k1 = []
            
            if type(k1) is tuple:
                for i in range(len(k1)):
                    z_k1.append(z[i]+dt/2*k1[i])
                z_k1 = tuple(z_k1)

                k2 = func(z_k1,t+dt/2)
                z_k2 = []
                for i in range(len(k2)):
                    z_k2.append(z[i]+dt/2*k2[i])
                z_k2 = tuple(z_k2)

                k3 = func(z_k2,t+dt/2)
                z_k3 = []
                for i in range(len(k3)):
                    z_k3.append(z[i]+dt*k3[i])
                z_k3 = tuple(z_k3)

                k4 = func(z_k3,t+dt)
                for i in range(len(k4)):
                    z[i] += dt/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
            else:
                z_k1 = z+dt/2*k1

                k2 = func(z_k1,t+dt/2)
                z_k2 = z+dt/2*k2

                k3 = func(z_k2,t+dt/2)
                z_k3 = z+dt*k3

                k4 = func(z_k3,t+dt)

                z += dt/6*(k1+2*k2+2*k3+k4)
            
            t += dt

    return z


def flat_parameters(parameters):
    theta_shape = []
    theta_flatten = []
    for p in parameters:
        theta_shape.append(p.size())
        theta_flatten.append(p.flatten())
    theta_flatten = torch.cat(theta_flatten)

    return theta_flatten

class AdjointFunc(torch.autograd.Function):
    def __init__(self):
        super(AdjointFunc,self).__init__()

    @staticmethod
    def forward(ctx,z0,func,t0,t1,theta_flatten):
        z1 = odeint(z0,func,t0,t1)
        ctx.func = func
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.save_for_backward(z1.clone())
        return z1
    
    @staticmethod
    def backward(ctx,dLdz1):
        func = ctx.func
        t0 = ctx.t0
        t1 = ctx.t1
        z1= ctx.saved_tensors[0]

        theta_list = list(func.parameters())
        
        s0 = [z1.clone(),dLdz1.clone()]

        for theta_i in theta_list:
            s0.append(torch.zeros_like(theta_i))

        def aug_dynamics(x,t):
            z,a,dfdth_unused = x[0],x[1],x[2]
            z = torch.autograd.Variable(z.data,requires_grad=True)
            torch.set_grad_enabled(True)#important

            f = func(z,t)
            f.backward(a)

            adfdz = z.grad
            #z.grad.zero_()#不要

            adfdth_list = []
            for theta_i in theta_list:
                adfdth_list.append(-theta_i.grad)
                theta_i.grad.zero_()
            
            return (f, -adfdz, *adfdth_list)

        rlt = odeint(s0,aug_dynamics,t1,t0)
        #z0 = rlt[0]
        dLdz0 = rlt[1]
        dLdth0 = []
        for i in range(2,len(rlt)):
            dLdth0.append(rlt[i])
        dLdth0 = flat_parameters(dLdth0)

        return dLdz0,None,None,None,dLdth0