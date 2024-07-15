"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Rectangle,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface MtebChartProps {
  data: any;
}

export default function MtebChart({ data }: MtebChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart
        width={500}
        height={300}
        data={data}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="name"
          fontSize="10px"
          interval={0}
          angle={-45}
          textAnchor="end"
        />
        <YAxis fontSize={12} />
        <Tooltip
          contentStyle={{
            fontSize: "12px",
            backgroundColor: "rgba(255, 255, 255, 0.9)",
            border: "1px solid #f5f5f5",
            borderRadius: "4px",
            boxShadow: "0 2px 4px 0 rgba(0,0,0,0.1)",
          }}
        />

        <Legend
          formatter={(value) => {
            if (value === "evrn") return "ES+VS+RR_n";
            return value.toUpperCase();
          }}
          wrapperStyle={{
            paddingTop: "1rem",
            fontSize: "12px",
          }}
        />
        <Bar dataKey="es" fill="#519DE9" />
        <Bar dataKey="vs" fill="#7CC674" />
        <Bar dataKey="ES+VS+RR_n" fill="#F4C145" />
      </BarChart>
    </ResponsiveContainer>
  );
}
