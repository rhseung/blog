**문제 링크**
[10908번: Phibonacci](https://www.acmicpc.net/problem/10908)

**사용한 알고리즘**
- [[수학]]
- [[분할 정복을 이용한 거듭제곱]]

**문제 설명**
$$
F_{n} = \dfrac{\phi^n - (1-\phi)^n}{\sqrt{ 5 }} \quad \text{where} \; \phi = \dfrac{\sqrt{ 5 } + 1}{2}
$$
$P_{0}=1, \, P_{1}=\phi,\, P_{n} = P_{n-1} + P_{n-2}$ 인 `Phibonacci`라는 수열이 있다고 할 때, 편의 상 $F_{-1} = 1$이라고 하면, $P_{n}=F_{n}\phi + F_{n-1} \; (n \ge 0)$ 로 나타낼 수 있다. 이제, $(P_n)^k$를 $A\phi^k + B$ 꼴로 표현할 수 있다면 그 상수를 구하고 아니면 -1을 출력하라.

**문제 풀이**
일단 `Phibonacci` 수열의 일반항을 구해볼까? 특성 방정식을 사용해서 $x^2 - \phi x - 1 = 0$의 근을 구하면, $x = \dfrac{\phi \pm \sqrt{ \phi^2 + 4 }}{2}$.
별로 소득이 없어 보인다...

$\phi$는 이 문제에서 $x^2-x-1=0$의 근이므로, $\phi^2  = \phi + 1$이 성립한다. 이를 계속 적용해보면,
$\phi^3 = \phi^2+\phi=2\phi+1$
$\phi^4=\phi^3+\phi^2=(2\phi+1) + (\phi+1)=3\phi+2$,
$\phi^n=F_{n}\phi+F_{n-1}$이 성립함을 알 수 있다. (수학적 귀납법으로 증명 가능)

따라서, $(P_{n})^k=\phi^{nk}=F_{nk}\phi+F_{nk-1}$이다.

$\phi^{nk}=A\phi^k+B \iff F_{nk}\phi+F_{nk-1}=A(F_{k}\phi + F_{k-1}) + B$이려면, $\phi$는 무리수이므로 항등식을 풀어보면 된다.
$F_{nk}=AF_{k},\,F_{nk-1}=AF_{k-1}+B$.

이 때, 수학적 귀납법을 통해 $F_{k} \vert F_{nk}$인지 확인해보자.
$n=1$일 때, 자명하게 성립.
$n=\alpha$일 때, $F_{k} \vert F_{\alpha k}$가 성립한다고 가정.
$n=\alpha + 1$일 때, $F_{(\alpha + 1)k}=F_{\alpha k}F_{k-1} + F_{\alpha k + 1}F_{k}$이다. 이는 [증명](https://math.stackexchange.com/questions/11477/fibonacci-addition-law-f-nm-f-n-1f-m-f-n-f-m1?utm_source=chatgpt.com)을 참고하였다.
> $M^n=\begin{bmatrix}1 & 1 \\ 1 & 0\end{bmatrix}^n=\begin{bmatrix}F_{n-1} & F_{n} \\ F_{n} & F_{n+1}\end{bmatrix}$
> $M^{m+n}=M^mM^n=\begin{bmatrix}F_{n+1}F_{m+1}+F_{n}F_{m} & F_{n+1}F_{m} + F_{n}F_{m-1} \\ F_{n}F_{m+1}+F_{n-1}F_{m} & F_{n}F_{m}+F_{n-1}F_{m-1}\end{bmatrix}$
> $\therefore F_{n+m}=F_{n+1}F_{m}+F_{n}F_{m-1}$
> 이를 사용해, $n=\alpha k$, $m=k$로 보면 위의 식이 성립한다.

즉, $F_{k} \vert F_{(\alpha+1) k}$가 성립하므로, 모든 자연수에 대해 $F_{k} \vert F_{nk}$이다.
이는, $A,\, B$가 항상 존재함을 시사한다. 따라서 답이 -1이 되는 경우는 없을 것이다.

그럼, 값은 어떻게 구할까? 문제에서 $10^9+7$에 대한 모듈러를 묻고 있으므로 페르마 소정리로 $F_{k}$의 모듈러 역원을 구해 생각보다 쉽게 $A=\dfrac{F_{nk}}{F_{k}}$를 구할 수 있다. $B$는 뭐 그냥 빼면 되고.

피보나치는 행렬 + 고속 제곱해서 구하면 된다.

---

여기서 제출하면 WA를 받았는데, 이는 모듈러 역원을 `pow(f_k, -1, mod)`로 구할 때, $\gcd(f_{k}, mod) \neq 1$일 때 `ValueError`를 받게 된다. 그리고 $F_{-1}=1$을 빼먹은 것도 덤.

[\[BOJ 10908\] Phibonacci](https://hapby9921.tistory.com/entry/BOJ-10908-Phibonacci)를 참고하여 $p^2$으로 하는 테크닉을 사용해서 AC를 받았으나 이해가 부족하다.

**소스코드**
```python
# BOJ 10908 - Phibonacci  
import sys  
  
input = sys.stdin.readline  
  
mod = 1000000007  
  
def mul_mod(a: int, b: int, MOD: int) -> int:  
    return (a % MOD) * (b % MOD) % MOD  
  
  
def mul_2x2_mod(mat1: list[list[int]], mat2: list[list[int]], MOD: int) -> list[list[int]]:  
    assert len(mat1) == 2 and len(mat1[0]) == 2  
    assert len(mat2) == 2 and len(mat2[0]) == 2  
  
    a = mat1[0][0]; b = mat1[0][1]  
    c = mat1[1][0]; d = mat1[1][1]  
  
    e = mat2[0][0]; f = mat2[0][1]  
    g = mat2[1][0]; h = mat2[1][1]  
  
    return [  
        [(mul_mod(a, e, MOD) + mul_mod(b, g, MOD)) % MOD, (mul_mod(a, f, MOD) + mul_mod(b, h, MOD)) % MOD],  
        [(mul_mod(c, e, MOD) + mul_mod(d, g, MOD)) % MOD, (mul_mod(c, f, MOD) + mul_mod(d, h, MOD)) % MOD],  
    ]  
  
  
def fast_pow_mod(base: list[list[int]], exp: int, MOD: int) -> list[list[int]]:  
    if exp == 0:  
        return [[1, 0], [0, 1]]  
    if exp == 1:  
        return [[base[0][0] % MOD, base[0][1] % MOD], [base[1][0] % MOD, base[1][1] % MOD]]  
  
    half = fast_pow_mod(base, exp // 2, MOD)  
    remain = base if exp % 2 == 1 else [[1, 0], [0, 1]]  
    return mul_2x2_mod(mul_2x2_mod(half, half, MOD), remain, MOD)  
  
  
def fibonacci_mod(n: int, MOD: int) -> tuple[int, int]:  
    if n == 0:  
        return 0, 1  
    if n == 1:  
        return 1, 0  
    M_n = fast_pow_mod([[1, 1], [1, 0]], n - 1, MOD)  
    return M_n[0][0] % MOD, M_n[1][0] % MOD  
  
  
def main():  
    n, k = map(int, input().split())  
  
    mod_2 = mod * mod  
    f_nk, f_nk_1 = fibonacci_mod(n * k, mod_2)  
    f_k, f_k_1 = fibonacci_mod(k, mod_2)  
  
    if f_k % mod == 0:  
        f_nk //= mod  
        f_k //= mod  
  
    a = f_nk * pow(f_k, -1, mod) % mod  
    b = (f_nk_1 - a * f_k_1) % mod  
  
    print(a, b)

  
if __name__ == "__main__":  
    main()
```