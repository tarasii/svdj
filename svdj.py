'''
SVDJ Singular value decomposition using Jacobi algorithm.
  u, s, v = svdj(a)
Was adapted from Nicolas Le Bihan and Stephen J. Sangwine svdj matlab function
http://freesourcecode.net/matlabprojects/5649/sourcecode/svdj.m#.WyokgfZuKUk
'''

import numpy as np

def svdj(a, n_it=4):
	#tol = 2.2204e-016
	tol = 1.0e-014

	m, n = np.shape(a)
	r_c = [[r, c] for r in range(n-1) for c in range (r+1, n)]
                                            # F = str2func(class(A));
	v = np.eye(n, dtype=type(a[0, 0]))  # V = F(eye(N));
	
	on = sum(sum(abs(a)**2))/n  # On = 0; for c = A, On = On + sum(abs(c).^2); end; On = On ./ N;

	for i in range(n_it):
		cnt = 0
		for r, c in r_c:
			a1 = a[:, r]
			a2 = a[:, c]

			b_rr = sum(abs(a1)**2)          # b_rr = sum(abs(A(:,r)).^2);
			b_cc = sum(abs(a2)**2)          # b_cc = sum(abs(A(:,c)).^2);
			b_rc = np.dot(a1.conj().T , a2) # b_rc = A(:,r)' * A(:,c);
			m_b_rc = abs(b_rc);             # m = abs(b_rc);

			if not m_b_rc:
				continue

			tau_m = (b_cc - b_rr) / (2 * m_b_rc)   # tau = (b_cc - b_rr) / (2 * m);

			if not tau_m:
				continue

			cnt += 1
			t = np.sign(tau_m) / (abs(tau_m) + np.sqrt(1.0 + tau_m**2))  # t = sign(tau) ./ (abs(tau) + sqrt(1 + tau .^ 2));
			c_t = 1 / np.sqrt(1.0 + t**2)                           # C = 1 ./ sqrt(1 + t .^ 2); 
			s = b_rc * t * c_t / m_b_rc                             # S = (b_rc .* t .* C) ./ m;
			
			g = np.eye(n, dtype=type(a[0, 0]))                     # G = F(eye(N));
			g[r, r] = c_t                                          # G(r,r) = F(C);
			g[c, c] = c_t                                          # G(c,c) = F(C);
			g[r, c] = s                                            # G(r,c) = S;
			g[c, r] = -np.conj(s)                                  # G(c,r) =-conj(S);

			a = np.dot(a, g)                                       # A = A * G;
			v = np.dot(v, g)                                       # V = V * G;

		if cnt == 0:
			raise Exception('No rotations performed during sweep.')

		b = np.dot(a.conj().T, a)               # B = A' * A;
		off = sum(sum(abs(np.triu(b, 1))**2))/n	# Off = sum(sum(abs(triu(B, 1))))/(N.^2);

		if off/on < tol:
			break

	b = np.dot(a.conj().T, a)      # B = A' * A;
	t_ = np.sqrt(abs(np.diag(b)))  # T = sqrt(abs(diag(B)));
	ind = np.argsort(t_)[::-1]     # [T,IX] = sort(T);
	t_ = t_[ind]
	a = a[:,ind]                   # A = A(:, IX);
	v = v[:,ind]                   # V = V(:, IX);
	u = a / np.tile(t_.T, (m, 1))  # U = A ./ repmat(T',M,1);
	#s = np.diag(t_)               
	s = t_                         # S = diag(T); 

	return (u, s, v)


if __name__ == '__main__':
	a = np.array([[  -2125 + 1467*1j,    964 - 7903*1j ],
	              [   2818 - 2029*1j,  -9697 - 6936*1j ],
	              [   1956 - 7418*1j,    703 -   49*1j ]])
	
	print("a", a, a.shape)
	
	u, s, v = svdj(a)
	
	print("U", u)
	print("S", s)
	print("V", v)
	
	print("SVDJ U matrix don't match standart numpy SVD U matrix, S and V are close:")
	
	u, s, vh = np.linalg.svd(a, full_matrices=False)
	
	print("U", u)
	print("S", s)
	print("V", v)
