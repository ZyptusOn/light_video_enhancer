using Microsoft.UI.Xaml;
using Windows.Graphics;

namespace LightVideoEnhancer_WinUI;

public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.SetIcon("Assets/AppIcon.ico");
        AppWindow.Resize(new SizeInt32(1280, 900));
        RootFrame.Navigate(typeof(MainPage));
    }
}
