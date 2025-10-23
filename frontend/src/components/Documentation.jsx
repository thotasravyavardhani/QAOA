import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { BookOpen, Code, GitBranch, Zap } from 'lucide-react';

export default function Documentation() {
  return (
    <div className="space-y-6">
      {/* Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="w-6 h-6" />
            QAOA Overview
          </CardTitle>
          <CardDescription>
            Quantum Approximate Optimization Algorithm for NP-Hard Combinatorial Problems
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h3 className="font-semibold text-lg mb-2">What is QAOA?</h3>
            <p className="text-gray-700 leading-relaxed">
              The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-classical algorithm 
              designed to solve combinatorial optimization problems. It leverages quantum superposition and 
              entanglement to explore solution spaces more efficiently than classical algorithms.
            </p>
          </div>

          <div>
            <h3 className="font-semibold text-lg mb-2">Mathematical Formulation</h3>
            <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm space-y-2">
              <p>|ψ(β, γ)⟩ = U(B, βₚ)U(C, γₚ)...U(B, β₁)U(C, γ₁)|s⟩</p>
              <p className="text-xs text-gray-600 mt-2">where:</p>
              <ul className="text-xs text-gray-600 space-y-1 ml-4">
                <li>• U(C, γ) = e^(-iγC) is the cost unitary</li>
                <li>• U(B, β) = e^(-iβB) is the mixer unitary</li>
                <li>• C is the problem Hamiltonian</li>
                <li>• B is the mixer Hamiltonian</li>
                <li>• p is the number of QAOA layers (circuit depth)</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Problems */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitBranch className="w-6 h-6" />
            Supported Problems
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold">Max-Cut</h4>
                <Badge>Implemented</Badge>
              </div>
              <p className="text-sm text-gray-600">
                Partition graph vertices to maximize edges between sets. Applications in network design, 
                clustering, and VLSI design.
              </p>
            </div>

            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold">TSP</h4>
                <Badge variant="secondary">Backend Ready</Badge>
              </div>
              <p className="text-sm text-gray-600">
                Find shortest route visiting all cities. Classic optimization problem with applications 
                in logistics and route planning.
              </p>
            </div>

            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold">VRP</h4>
                <Badge variant="secondary">Backend Ready</Badge>
              </div>
              <p className="text-sm text-gray-600">
                Optimize routes for multiple vehicles with capacity constraints. Used in delivery 
                and distribution systems.
              </p>
            </div>

            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold">Graph Coloring</h4>
                <Badge variant="secondary">Backend Ready</Badge>
              </div>
              <p className="text-sm text-gray-600">
                Assign colors such that no adjacent vertices share colors. Applications in scheduling, 
                register allocation, and frequency assignment.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Features */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-6 h-6" />
            Key Features
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-3xl font-bold text-blue-600 mb-2">1-5</div>
              <div className="text-sm text-gray-600">QAOA Layers (p)</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-3xl font-bold text-green-600 mb-2">2-20</div>
              <div className="text-sm text-gray-600">Qubits Support</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-3xl font-bold text-purple-600 mb-2">NISQ</div>
              <div className="text-sm text-gray-600">Hardware Ready</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* API Usage */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="w-6 h-6" />
            API Usage Example
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`// Solve Max-Cut Problem
const response = await fetch('/api/maxcut/solve', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    num_vertices: 4,
    edges: [[0,1], [1,2], [2,3], [0,3]],
    p_layers: 2,
    max_iter: 100,
    method: 'COBYLA'
  })
});

const result = await response.json();
console.log('Cut Value:', result.cut_value);
console.log('Best Partition:', result.best_partition);`}
            </pre>
          </div>
        </CardContent>
      </Card>

      {/* References */}
      <Card>
        <CardHeader>
          <CardTitle>Research References</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm"</li>
            <li>2. Hadfield, S., et al. (2019). "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz"</li>
            <li>3. Crooks, G. E. (2018). "Performance of the Quantum Approximate Optimization Algorithm on the Maximum Cut Problem"</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
