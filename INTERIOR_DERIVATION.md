# Screen-center quotient VJP and analytic eligibility

Independent quotient derivation and reversed-full-chain control are carried
from Codex experiment172 (test branch ea70c61a58ea779765902ebdcf5204a9b03e1e42),
whose failed tile allocation and float32 scores remain negative evidence.
Base is PR6 d63fd31bfd822e1036456e089522aed18f4516a6, rasterizer
af4adebce2652a595fa98355d7b24890a72412da. Renderer/conditioning/backward unchanged.

Native contiguous pixel rows are u,v,w. For pixel q=(x,y), p=(xw-u)×(yw-v)
and s=p.xy/p.z, so rho3=s·s. With t=(9,9,-1), d=sum(t_j w_j²),
c_i=sum(t_j u_ij w_j)/d and rho2=2||q-c||². In the retained screen branch,
alpha=.7 exp(-||q-c||²). Incoming weights g=(.7,-.3) give
b_i=sum_q -2g_q alpha_q(c_i-q_i). Direct differentiation gives:

    bar_u_j = b_x t_j w_j/d
    bar_v_j = b_y t_j w_j/d
    bar_w_j = t_j [b_x(u_j-2c_x w_j)+b_y(v_j-2c_y w_j)]/d

At affine w=(0,0,1), four linear-block derivatives are exactly zero, translation
derivatives are b and three bottom-row derivatives generally nonzero. The mean
chain is Bᵀb, compared explicitly without autograd, plus actual producer VJP.
The 162 CPU native_backward has its own ray-plane VJP but uses autograd for
center preprocessing; it is a comparator, not the independent center proof.
Negating the complete 3×3 quotient VJP must fail atol3e-5/rtol3e-4 for every arm.

## Conservative eligibility before derivative scores

For each frozen base and endpoint, construct component-wise h boxes padded by
2e-5 (float32) or2e-12 (float64), much larger than unit roundoff for these values.
All interval arithmetic rounds endpoints outward with nextafter. Bound d away
from zero, compute the quotient center, cross-product p and p.xy/p.z interval,
then rho3 lower, rho2 upper, positive offset lower and alpha interval. This is
finite-endpoint analytic interval evaluation, not derivative scoring or a
parameter search. Output includes every pixel at every endpoint.

Use the exact affine predicate of each concrete endpoint. Affine support cutoff
R=sqrt(2log(255*.7)); coordinate extents R||row_i(A)||; radius=1+ceil(max(extent,
R*.707106)). Projective endpoints use cutoff3, extents
sqrt(max(1e-4,c_i²-sum_j(t_j h_ij²/d))) and ceil(max(extent,3*.707106)).
Interval upper extents plus the screen-filter lower radius enclose all possible
integer radii. Enumerate those radii and center interval endpoints in native
getRect: clamp(trunc((c-r)/16),0,2), clamp(trunc((c+r+15)/16),0,2).
Monotonicity makes this a conservative rectangle proof. All are(0,0,1,1).
Pixels lie within that rectangle. The exact actual tensor rectangles are also
checked, including every +/- producer mean and all nine matrix perturbations.

The prespecified bounds are exponent gap≥.05, cutoff margin≥.01, saturation
margin≥.1, no zero offset and identical actual rectangle. Bound values are in
outputs/analytic-eligibility-summary.json, committed full tables under fixtures.
The chosen roundoff boxes are conservative sensitivity enclosures, not a formal
proof of a compiled implementation's error; native execution still needs exact
source/build receipts. No CPU equation is relabeled as a compiled-kernel test.

Method context: Yang, Zhou and Huang, Introducing Unbiased Depth into2D Gaussian
Splatting, https://arxiv.org/abs/2503.06587 ;
https://github.com/gavingavinchan/Unbiased_Surfel_sonar ;
https://github.com/gavingavinchan/diff_surfel_rasterization .
