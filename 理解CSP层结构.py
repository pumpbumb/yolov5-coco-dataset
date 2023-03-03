import torch
import torch.nn as nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = (shortcut and c1 == c2)  # 只有 c1 == c2 , 也就是瓶颈结构的输入等于输出时，才有捷径连接

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
        # 上一行就是 self.m = Bottleneck(c_, c_, shortcut, g, e=1.0)
        # 疑问：self.m 属性的 e 不应该是 0.5嘛，怎么是 1.0

    def forward(self, x):
        return self.cv3(
            torch.cat(
                (self.m(self.cv1(x)), self.cv2(x)), dim=1)
        )


if __name__ == '__main__':
    # 实例化一个 CSP 类
    csp = C3(128, 64)
    print(csp.m)
    example = torch.rand(size=(16, 128, 160, 160))  # batch_size, channel_number, height, width
    out = csp(example)
    print(example.shape, out.shape, sep='\n', end='\n\n')

    # -------------------------------------------------------------------------------------
    # 理解 C3 类中的 self.m 属性
    # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    # -------------------------------------------------------------------------------------
    a1, a2, a3 = [2 for _ in range(3)]
    print(a1, a2, a3)
