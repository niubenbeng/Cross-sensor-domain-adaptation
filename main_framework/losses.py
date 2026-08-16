import math
import torch
import torch.nn as nn


class MMSD(nn.Module):
    """Multi-kernel RBF MMSD (squared-kernel V-statistic).

    Used as plain marginal alignment during warmup (JVM warmup = MMSD).
    """

    def __init__(self):
        super(MMSD, self).__init__()

    def _mix_rbf_mmsd(self, X, Y, sigmas=(1,), wts=None, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmsd(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX, dim1=-2, dim2=-1)
        Y_sqnorms = torch.diagonal(YY, dim1=-2, dim2=-1)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma ** 2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
            return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _mmsd(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = torch.tensor(K_XX.size(0), dtype=torch.float32)
        n = torch.tensor(K_YY.size(0), dtype=torch.float32)
        C_K_XX = torch.pow(K_XX, 2)
        C_K_YY = torch.pow(K_YY, 2)
        C_K_XY = torch.pow(K_XY, 2)
        if biased:
            mmsd = (torch.sum(C_K_XX) / (m * m) + torch.sum(C_K_YY) / (n * n)
            - 2 * torch.sum(C_K_XY) / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(C_K_XX)
                trace_Y = torch.trace(C_K_YY)

            mmsd = ((torch.sum(C_K_XX) - trace_X) / ((m - 1) * m)
                    + (torch.sum(C_K_YY) - trace_Y) / ((n - 1) * n)
                    - 2 * torch.sum(C_K_XY) / (m * n))
        return mmsd

    def forward(self, X1, X2, bandwidths=[3]):
        kernel_loss = self._mix_rbf_mmsd(X1, X2, sigmas=bandwidths)
        return kernel_loss


class JMMSD(nn.Module):
    """Joint MMSD: integrates joint distribution alignment into the MMSD squared-kernel framework.

    Joint kernel K_joint = K_feat * K_softmax, then MMSD squared-kernel V-statistic
    (C_K = K_joint^2, more sensitive to distribution differences).

    Two-layer activation (soft labels for stability):
        Layer 1: 128D features (pre-classifier representation) -- multi-kernel RBF MMSD
        Layer 2: 5D softmax output (classifier decision)      -- multi-kernel RBF MMSD
    """

    def __init__(self, bandwidths_feat=(3,), bandwidths_softmax=(3,)):
        super(JMMSD, self).__init__()
        self.mmsd = MMSD()
        self.bandwidths_feat = list(bandwidths_feat)
        self.bandwidths_softmax = list(bandwidths_softmax)

    def forward(self, feat1, feat2, output1, output2):
        softmax1 = torch.softmax(output1, dim=1)
        softmax2 = torch.softmax(output2, dim=1)
        Kf_XX, Kf_XY, Kf_YY, df = self.mmsd._mix_rbf_kernel(feat1, feat2, self.bandwidths_feat)
        Ks_XX, Ks_XY, Ks_YY, ds = self.mmsd._mix_rbf_kernel(softmax1, softmax2, self.bandwidths_softmax)
        K_XX = Kf_XX * Ks_XX
        K_XY = Kf_XY * Ks_XY
        K_YY = Kf_YY * Ks_YY
        return self.mmsd._mmsd(K_XX, K_XY, K_YY, const_diagonal=df * ds, biased=True)


class VDR(nn.Module):
    """VDR (centered squared-kernel V-statistic, Student-t kernel)."""

    def __init__(self, sigmas=(0.6,), wts=None, biased=True):
        super(VDR, self).__init__()
        self.sigmas = sigmas
        self.wts = wts if wts is not None else [1] * len(sigmas)
        self.biased = biased

    def forward(self, X, Y):
        return self.mix_student_vdr2(X, Y)

    def mix_student_vdr2(self, X, Y):
        K_XX, K_XY, K_YY, d = self._mix_student_kernel(X, Y)
        return self._vdr2(K_XX, K_XY, K_YY, const_diagonal=d)

    def _mix_student_kernel(self, X, Y):
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX, dim1=-2, dim2=-1)
        Y_sqnorms = torch.diagonal(YY, dim1=-2, dim2=-1)

        r = lambda x: x.unsqueeze(0)
        c = lambda x: x.unsqueeze(1)

        K_XX, K_XY, K_YY = 0, 0, 0
        for sigma, wt in zip(self.sigmas, self.wts):
            gamma = math.gamma((sigma + 1) / 2) / (math.gamma(sigma / 2) * (sigma ** 0.5))
            K_XX += wt * gamma * torch.pow((1. + (-2 * XX + c(X_sqnorms) + r(X_sqnorms)) / 2.), -(sigma + 1) / 2)
            K_XY += wt * gamma * torch.pow((1. + (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)) / 2.), -(sigma + 1) / 2)
            K_YY += wt * gamma * torch.pow((1. + (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)) / 2.), -(sigma + 1) / 2)

        return K_XX, K_XY, K_YY, torch.sum(torch.tensor(self.wts, dtype=torch.float32))

    def _vdr2(self, K_XX, K_XY, K_YY, const_diagonal=False):
        m = K_XX.size(0)
        n = K_YY.size(0)

        CM_m = torch.eye(m, device=K_XX.device) - (1 / m) * torch.ones((m, m), device=K_XX.device)
        CM_n = torch.eye(n, device=K_YY.device) - (1 / n) * torch.ones((n, n), device=K_YY.device)

        C_K_XX = torch.pow(torch.matmul(torch.matmul(CM_m, K_XX), CM_m.t()), 2)
        C_K_YY = torch.pow(torch.matmul(torch.matmul(CM_n, K_YY), CM_n.t()), 2)
        C_K_XY = torch.pow(torch.matmul(torch.matmul(CM_m, K_XY), CM_n.t()), 2)

        if self.biased:
            vdr2 = (torch.sum(C_K_XX) / (m * m) +
                    torch.sum(C_K_YY) / (n * n) -
                    2 * torch.sum(C_K_XY) / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(C_K_XX)
                trace_Y = torch.trace(C_K_YY)

            vdr2 = ((torch.sum(C_K_XX) - trace_X) / ((m - 1) * (m - 2)) +
                    (torch.sum(C_K_YY) - trace_Y) / ((n - 1) * (n - 2)) -
                    2 * torch.sum(C_K_XY) / ((m - 1) * (n - 1)))
        return vdr2


class JVDR(nn.Module):
    """Joint VDR: integrates joint distribution alignment into the VDR squared-kernel (centered) framework.

    Joint kernel K_joint = K_feat * K_softmax (Student-t kernel), then VDR centered squared-kernel
    V-statistic: C_K = (CM @ K_joint @ CM)^2.
    VDR centering is sensitive to distribution mean shift; Student-t heavy-tailed kernel is robust to outliers.
    """

    def __init__(self, sigmas_feat=(0.6,), sigmas_softmax=(0.6,), biased=True):
        super(JVDR, self).__init__()
        self.vdr_feat = VDR(sigmas=sigmas_feat, biased=biased)
        self.vdr_softmax = VDR(sigmas=sigmas_softmax, biased=biased)
        self.biased = biased

    def forward(self, feat1, feat2, output1, output2):
        softmax1 = torch.softmax(output1, dim=1)
        softmax2 = torch.softmax(output2, dim=1)
        Kf_XX, Kf_XY, Kf_YY, df = self.vdr_feat._mix_student_kernel(feat1, feat2)
        Ks_XX, Ks_XY, Ks_YY, ds = self.vdr_softmax._mix_student_kernel(softmax1, softmax2)
        K_XX = Kf_XX * Ks_XX
        K_XY = Kf_XY * Ks_XY
        K_YY = Kf_YY * Ks_YY
        return self.vdr_feat._vdr2(K_XX, K_XY, K_YY, const_diagonal=df * ds)


class JVM(nn.Module):
    """Joint MMSD-VDR Squared: fuses JMMSD (RBF + squared) and JVDR (Student-t + centered squared).

    The two joint alignment methods are complementary:
    - JMMSD (RBF Gaussian kernel + MMSD squared): local kernel, strong at aligning distribution shape
    - JVDR  (Student-t kernel + VDR centered squared): heavy-tailed kernel robust to outliers
    """

    def __init__(self, bandwidths_feat=(3,), bandwidths_softmax=(3,),
                 sigmas_feat=(0.6,), sigmas_softmax=(0.6,),
                 w_v=1.0, w_m=1.0):
        super(JVM, self).__init__()
        self.jmmsd = JMMSD(bandwidths_feat=bandwidths_feat,
                           bandwidths_softmax=bandwidths_softmax)
        self.jvdr = JVDR(sigmas_feat=sigmas_feat,
                         sigmas_softmax=sigmas_softmax)
        self.w_v = w_v
        self.w_m = w_m

    def forward(self, feat1, feat2, output1, output2):
        loss_mmsd = self.jmmsd(feat1, feat2, output1, output2)
        loss_vdr = self.jvdr(feat1, feat2, output1, output2)
        return self.w_v * loss_vdr + self.w_m * loss_mmsd
