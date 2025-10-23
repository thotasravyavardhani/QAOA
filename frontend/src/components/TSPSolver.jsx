import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Construction } from 'lucide-react';

export default function TSPSolver() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Traveling Salesman Problem (TSP)</CardTitle>
        <CardDescription>Find the shortest route visiting all cities</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-12 text-gray-400">
          <Construction className="w-16 h-16 mx-auto mb-4" />
          <p>TSP Solver - Similar to Max-Cut implementation</p>
          <p className="text-sm mt-2">Configure distance matrix and solve with QAOA</p>
        </div>
      </CardContent>
    </Card>
  );
}
