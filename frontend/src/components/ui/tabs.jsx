import React from 'react';

export function Tabs({ className = '', children, value, onValueChange, ...props }) {
  return (
    <div className={`${className}`} {...props} data-value={value}>
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child, { value, onValueChange });
        }
        return child;
      })}
    </div>
  );
}

export function TabsList({ className = '', children, value, onValueChange, ...props }) {
  return (
    <div
      className={`inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground ${className}`}
      {...props}
    >
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child, { value, onValueChange });
        }
        return child;
      })}
    </div>
  );
}

export function TabsTrigger({ className = '', children, value: triggerValue, value: currentValue, onValueChange, ...props }) {
  const isActive = currentValue === triggerValue;
  
  return (
    <button
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50 ${
        isActive ? 'bg-background text-foreground shadow-sm' : 'hover:bg-gray-100'
      } ${className}`}
      onClick={() => onValueChange && onValueChange(triggerValue)}
      {...props}
    >
      {children}
    </button>
  );
}

export function TabsContent({ className = '', children, value: contentValue, value: currentValue, ...props }) {
  if (contentValue !== currentValue) return null;
  
  return (
    <div
      className={`mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
